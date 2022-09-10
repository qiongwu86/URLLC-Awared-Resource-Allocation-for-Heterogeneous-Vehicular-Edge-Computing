import numpy as np
#import tensorflow as tf
from collections import OrderedDict, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
from nn_function.pythonplusplus import identity
from nn_function import pytorch_util as ptu
from torch.distributions import MultivariateNormal
from nn_function.distributions import TanhNormal
from nn_function.core import PyTorchModule
from nn_function.normalization import LayerNorm
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SACLosses = namedtuple(
    'SACLosses',
    'policy_loss qf1_loss qf2_loss alpha_loss',
)

class Mlp(PyTorchModule):
    def __init__(
            self,
            hidden_sizes,       # [256,256]
            output_size,        # =1
            input_size,         # =action_dim+obs_dim = 13+6
            init_w=3e-3,        # 当成
            hidden_activation=F.relu,   #
            output_activation=identity, # 恒同映射
            hidden_init=ptu.fanin_init, # return tensor.data.uniform_(-bound, bound) 产生(-bound, bound)均匀分布
            b_init_value=0.,
            layer_norm=False,           # 网络层 norm 化
            layer_norm_kwargs=None,
    ):
        super().__init__()
        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()
        self.input_size = input_size                # 23 因为输出的是 action 和 state
        self.output_size = output_size
        self.hidden_activation = hidden_activation  # 这里没有任何输入参数，说明附值为一个函数 F.relu
        self.output_activation = output_activation  # 这里没有任何输入参数，说明附值为一个函数 identity
        self.layer_norm = layer_norm                # none
        self.fcs = []
        self.layer_norms = []
        in_size = input_size
        for i, next_size in enumerate(hidden_sizes): # 0 256 ; 1 256
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)              # fc.weight 网络的权重 。 均匀分布权重。 但函数没有接收返回，是怎么改变fc.weight的值的？
            fc.bias.data.fill_(b_init_value)    #  bias  参数初始化 = 0
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)
            if self.layer_norm:                 # 网络层 norm 化
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.fill_(0)
    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []  # action_logprob
        self.rewards = []
        self.next_states = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.next_states[:]

class ActorCritic(nn.Module):
    def __init__(self,n,hidden_init=ptu.fanin_init,b_init_value=0.,init_w=3e-3):
        super(ActorCritic, self).__init__()

        self.hidden1 = nn.Linear(48, 256)  # 输入参数 5 待修改
        hidden_init(self.hidden1.weight)
        self.hidden1.bias.data.fill_(b_init_value)
        self.hidden2= nn.Linear(256, 256)  # 输入参数 5 待修改
        hidden_init(self.hidden2.weight)
        self.hidden2.bias.data.fill_(b_init_value)


        self.last_mean = nn.Linear(256, 4+9)
        self.last_mean.weight.data.uniform_(-init_w, init_w)
        self.last_mean.bias.data.fill_(0)

        self.last_fc_log_std = nn.Linear(256, 4+9)  # 输出层
        self.last_fc_log_std.weight.data.uniform_(-init_w/3, init_w/3)  # 输出层权重初始化
        self.last_fc_log_std.bias.data.uniform_(-init_w/3, init_w/3)  # 输出层bias初始化
    def forward(self, state, memory):
        raise NotImplementedError

    def data_process(self,state):
        #state =(state-state.mean(axis=1))/(state.std(axis=1)+1e-20)
        #state1 = (state-state.min(axis=1))/(state.max(axis=1)-state.min(axis=1)+1e-20)
        state1 = state.copy()
        return state1

    def act_dist(self,state):
        # 这里的一开始state是列表的格式
        state1 = state.reshape(len(state),1,8,-1)
        state1 = np.transpose(state1, (0,1,3,2))

        state = torch.FloatTensor(state1[:,0,:,:]).to(device)

        h = F.relu(self.hidden1(state.reshape(len(state), 1, -1)))
        h = F.relu(self.hidden2(h.reshape(len(state), 1, -1)))

        mean=self.last_mean(h.reshape(len(state), 1, -1)) # choose : [1,4], exercise : [256,6]
        log_std = self.last_fc_log_std(h.reshape(len(state), 1, -1))
        log_std = torch.clamp(log_std, -20, 2)        # log_std本来就是在-20到20之间，夹逼也没什么作用
        std = torch.exp(log_std)                                    # choose : [1,4] , exercise :  [256,6]

        mean = mean[:,0,:]
        std = std[:,0,:]
        return TanhNormal(mean, std)

    def act(self,state,memory):
        state = np.hstack((state[0:2*8],state[-4*8:]))
        state = self.data_process(np.array(state.reshape(1,-1)))
        dist = self.act_dist(state)  # -->list():  [[]]
        actions = dist.sample()
        #print("神经网络输出action ： ",actions)
        state = torch.FloatTensor(state).to(device) # memory存储state的格式为tensor格式
        memory.states.append(state)             # tensor([[]])
        memory.actions.append(actions)
        "这里得到改吧"
        memory.next_states.append(state)
        return actions

    def evaluate(self,state, action):
        action_mean = self.act_mean(state.cpu().data.numpy()) # -->np.array: [[],[],[]]
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        dist = MultivariateNormal(action_mean, cov_mat)
        action_logprobs = dist.log_prob(action)  # 相对熵 对数概率
        dist_entropy = dist.entropy()  # 熵
        state_value = self.critic(state)
        return action_logprobs, torch.squeeze(state_value), dist_entropy
class ConcatMlp(Mlp):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    def __init__(self, *args, dim=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=self.dim)   # self.dim = 1
        return super().forward(flat_inputs, **kwargs)
class DRL_actor:
    def __init__(self, f_E,epsilon,gamma,n):
        self.action_dimension=n
        self.f_E = f_E
        self.fail_epsilon = epsilon
        self.epsilon_max = 0.9
        self.gamma = gamma
        self.K_epochs = 80       #
        self.eps_clip = 0.2      # 0.2
        self.c1 = 0.01
        self.c2 = 1
        self.batch_size = 256
        self.policy = ActorCritic(n=self.action_dimension).to(device) # 创建 action_std =0.5 作用分布的常数标准差（多元正态
        self.use_automatic_entropy_tuning = True  # True
        self.soft_target_tau = 0.005   # 0.005: 用于target 网络更新参数
        self.qf1 = ConcatMlp(input_size=48+13,output_size=1,hidden_sizes=[256, 256],)
        self.qf2 = ConcatMlp(input_size=48+13,output_size=1,hidden_sizes=[256, 256],)
        self.target_qf1 = ConcatMlp(input_size=48+13, output_size=1, hidden_sizes=[256, 256], )
        self.target_qf2 = ConcatMlp(input_size=48+13, output_size=1, hidden_sizes=[256, 256], )
        self.train()
    def choose_action(self,state,memory):
        action = self.policy.act(state,memory)
        action = action / 2 ;
        action_comp = torch.softmax(action[0][:4].reshape(1, -1),dim=1).cpu().data.numpy().flatten()

        action_comm = action[:, self.action_dimension + 1:]
        for  i in range(self.action_dimension):
            action_comm[:,i*self.action_dimension:i*self.action_dimension+self.action_dimension]= torch.softmax(action_comm[:,i*3:i*3+3].reshape(1, -1),dim=1).cpu()
        action_comm = action_comm.data.numpy().flatten()
        return action_comp.reshape(1,-1),action_comm.reshape(self.action_dimension,3)

    def train(self,policy_lr=1e-4,qf_lr=1e-3,optimizer_class = optim.Adam,):
        self.target_update_period = 1  # target 网络更新参数间隔
        self.target_entropy = None
        if self.use_automatic_entropy_tuning:
            if self.target_entropy is None:
                # Use heuristic value from SAC paper
                # 使用SAC论文中的启发值
                self.target_entropy = -np.prod( 13                     # 连乘操作---维度相乘 = 6 : 这个数应该就是论文里面的 H = dim(a)
                    ).item()                                          # 动作为6维, 每个action取值为 -1 和 1 = Box([-1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1.], (6,), float32)
            else:
                self.target_entropy = set.target_entropy
        self.log_alpha = ptu.zeros(1, requires_grad=True)       # self.log_alpha = tensor([0.],requires_grad = True)
        self.alpha_optimizer = optimizer_class( [self.log_alpha],lr=policy_lr,)
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.policy_optimizer = optimizer_class(self.policy.parameters(),lr=policy_lr,)
        self.qf1_optimizer = optimizer_class(self.qf1.parameters(),lr=qf_lr,)
        self.qf2_optimizer = optimizer_class(self.qf2.parameters(),lr=qf_lr,)
        self.discount = 0.99  # 0.99
        self.reward_scale = 1  # 1
        self._n_train_steps_total = 0 # self._n_train_steps_total % self.target_update_period 更新target

    def compute_loss( self,s,r,a,s_):
        dist = self.policy.act_dist(s)
        new_obs_actions, log_pi = dist.rsample_and_logprob()  # new_obs_actions [256,4] , log_pi= {Size:1} 256
        log_pi = log_pi.unsqueeze(-1)  # log_pi : log(pi(a|s)) 变成 2 维
        if self.use_automatic_entropy_tuning:
            # 为什么这里是 self.log_alpha = 0(初始值) ，论文里是 alpha ？？？？？？？？？？？？？
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean() # self.target_entropy = dim(a) < 0 ?
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1
        q_new_actions = torch.min(                   # --> [256,1]
            #self.qf1(s[:,:32], new_obs_actions),     # Q1(st,at+1)
            #self.qf2(s[:,:32], new_obs_actions),     # Q2(st,at+1)
            self.qf1(s, new_obs_actions),     # Q1(st,at+1)
            self.qf2(s, new_obs_actions),     # Q2(st,at+1)
        )
        policy_loss = (alpha*log_pi - q_new_actions).mean() # 注意取均值 。 q_new_actions 为什么是下一个状态 new_obs_action
        """
        QF Loss
        """
        # 用来产生 t+1 时刻的 Q 值，需要用到 t+1 时刻的 pi

        q1_pred = self.qf1(s, a)                 # Q1(st,at) --> [256,1]
        q2_pred = self.qf2(s, a)                 # Q2(st,at) --> [256,1]
        next_dist = self.policy.act_dist(s_)     # s(t+1) 时刻的 pi ----> 产生t+1时刻的at+1
        new_next_actions, new_log_pi = next_dist.rsample_and_logprob() # new_next_actions: [256,4] , new_log_pi = {Size:1} 256
        new_log_pi = new_log_pi.unsqueeze(-1)     # 变成 2 维 : new_log_pi = {Size:1} 256--> new_log_pi = {Size:2} [256,1]
        target_q_values = torch.min(
            self.target_qf1(s_, new_next_actions),  # s(t+1),a(t+1)时刻的Q
            self.target_qf2(s_, new_next_actions),
        ) - alpha * new_log_pi
        "self.reward_scale是什么？？？？？？？？？"
        q_target = self.reward_scale * r.unsqueeze(-1).to(torch.float32) + self.discount * target_q_values  # r = {Size:2}[256,1]
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach()) # qf_criterion = nn.MSELoss()
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())
        loss = SACLosses(
            policy_loss=policy_loss,
            qf1_loss=qf1_loss,
            qf2_loss=qf2_loss,
            alpha_loss=alpha_loss,
        )
        # print(" A loss : ",policy_loss)
        # print("Q1 loss : ", policy_loss)
        # print("Q2 loss : ", policy_loss)
        return loss

    def train_from_torch(self, s,r,a,s_):
        losses = self.compute_loss(s,r,a,s_)
        """
        Update networks
        """
        # print("actor loss : " , losses.policy_loss)
        # print("  qf1_loss : ", losses.qf1_loss)
        # print("  qf2_loss : ", losses.qf2_loss)

        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            losses.alpha_loss.backward()
            self.alpha_optimizer.step()

        self.policy_optimizer.zero_grad()
        losses.policy_loss.backward()
        self.policy_optimizer.step()

        self.qf1_optimizer.zero_grad()
        losses.qf1_loss.backward()
        self.qf1_optimizer.step()
        #print(self.qf1.state_dict())

        self.qf2_optimizer.zero_grad()
        losses.qf2_loss.backward()
        self.qf2_optimizer.step()

        self._n_train_steps_total += 1
        self.try_update_target_networks()

        self.loss = losses
        #print("aplha loss : ",losses.alpha_loss,"alpha : ",self.log_alpha.exp())
    def try_update_target_networks(self):
        if self._n_train_steps_total % self.target_update_period == 0:
            self.update_target_networks()
    def update_target_networks(self):
        ptu.soft_update_from_to(
            self.qf1, self.target_qf1, self.soft_target_tau
        )
        ptu.soft_update_from_to(
            self.qf2, self.target_qf2, self.soft_target_tau
        )

    def update(self, memory, c1=0.01, c2=1):
        rewards = []
        discounted_reward = 0
        rewards = torch.tensor(memory.rewards[:999]).to(device)
        #rewards = (rewards - rewards.min()) / (rewards.max() - rewards.min() + 1e-10) #
        #rewards = (rewards - rewards.mean()) / (rewards.std()+ 1e-10)  #
        old_states = torch.squeeze(torch.stack(memory.states[:999]).to(device)).detach() #
        old_actions = torch.squeeze(torch.stack(memory.actions[:999]).to(device)).detach() #
        next_states = torch.squeeze(torch.stack(memory.next_states[1:]).to(device)).detach()
        for _ in range(self.K_epochs): # 80 次训练更新
            indices = np.random.choice(999, size=256, replace=False) # #True表示可以取相同数字，False表示不可以取相同数字
            # s = old_states[indices]
            s = old_states[indices]
            r = rewards[indices]
            a = old_actions[indices]
            s_= next_states[indices]
            self.train_from_torch(s,r,a,s_)
            # action_mean = self.act_mean(state.cpu().data.numpy())  # -->np.array: [[],[],[]]
        return self.log_alpha.exp(),self.loss

    def save(self, filename, directory):
        torch.save(self.policy.state_dict(), '{}/{}_policy_network.pth'.format(directory, filename))
        torch.save(self.qf1.state_dict(), '{}/{}_qf1_network.pth'.format(directory, filename))
        torch.save(self.qf2.state_dict(), '{}/{}_qf2_network.pth'.format(directory, filename))
        torch.save(self.target_qf1.state_dict(), '{}/{}_target_qf1_network.pth'.format(directory, filename))
        torch.save(self.target_qf2.state_dict(), '{}/{}_target_qf2_network.pth'.format(directory, filename))

    def load(self, filename, directory):
        self.policy.load_state_dict(torch.load('{}/{}_policy_network.pth'.format(directory, filename)))
        self.qf1.load_state_dict(torch.load('{}/{}_qf1_network.pth'.format(directory, filename)))
        self.qf2.load_state_dict(torch.load('{}/{}_qf2_network.pth'.format(directory, filename)))
        self.target_qf1.load_state_dict(torch.load('{}/{}_target_qf1_network.pth'.format(directory, filename)))
        self.target_qf1.load_state_dict(torch.load('{}/{}_target_qf2_network.pth'.format(directory, filename)))










