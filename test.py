from constants import *
from DRL_SAC import DRL_actor
from DRL_SAC import *
import Env_SAC as environment
#from DRL import DRL_actor
import time
import matplotlib.pyplot as plt
dir = "C:\\Users\\Administrator\\Desktop\\URLLC-Awared Resource Allocation for Heterogeneous Vehicular Edge Computing"
#model_dir = "\save_data2022.4.11\save_mode"
#save_dir  = "\save_data2022.4.11\\test_data"
model_dir = "\\save_mode"
save_dir  = "\\Data"
V = [0,10,50,100,200,500,1000]
V = [0,10,50,100]
Length  = []
Latency = []
T = [i for i in range(1,16001)]
action_v =[]
# Para  = []
for v in V:
    applications = (PROC_3D_GAME, VR, AR)
    task_rate = 5
    edge_capability = 60000 * GHZ
    violation_probability = 0.01
    Dicount_Factor = 0.9
    Model = "DRL"
    env = environment.MEC_v1(task_rate, edge_capability, 0, *applications, )
    env.V = v
    M = 5
    mmwave_gamma = np.random.gamma(M, 1 / M, 1)
    env.mmwave_gamma = mmwave_gamma[0]
    actor = DRL_actor(edge_capability, violation_probability, Dicount_Factor, len(applications))
    memory = Memory()
    state = env.init(*applications)
    time_step = 0
    update_timestep = 1000
    actor.load('SAC_v_{}'.format(v),directory=dir+model_dir)
    # para = {}
    # for name, parameters in actor.policy.named_parameters():
    #     para[name] = parameters
    # Para.append(para)
    length1  = []
    length2  = []
    length3  = []
    delay_1 = []
    delay_2 = []
    delay_3 = []
    Utility = []
    action_1 = []
    action_2 = []
    action_3 = []
    action_sum = []
    for t in T:
        state, _, r, utility,delay, action_comm,action= env.step(actor, memory, Model)
        action_1.append( action[0][0])
        action_sum.append(sum(action[0][0:3]))
        S = state.reshape(8, 8)
        length1.append(S[1][5])
        length2.append(S[1][6])
        length3.append(S[1][7])
        delay_1.append(max(delay[6]["mmwave"][0], delay[6]["DSRC"][0], delay[6]["C-V2I"][0]))
        delay_2.append(max(delay[7]["mmwave"][0], delay[7]["DSRC"][0], delay[7]["C-V2I"][0]))
        delay_3.append(max(delay[8]["mmwave"][0], delay[8]["DSRC"][0], delay[8]["C-V2I"][0]))
        Utility.append(utility)

    # plt.plot(action_1)
    # plt.show()
    # plt.plot(length1)
    # plt.show()
    # np.save(dir+save_dir + "\\v_{}_delay1.npy".format(v), delay_1)
    # np.save(dir+save_dir + "\\v_{}_delay2.npy".format(v), delay_2)
    # np.save(dir+save_dir + "\\v_{}_delay3.npy".format(v), delay_3)
    #
    # np.save(dir+save_dir + "\\v_{}_length1.npy".format(v), length1)
    # np.save(dir+save_dir + "\\v_{}_length2.npy".format(v), length2)
    # np.save(dir+save_dir + "\\v_{}_length3.npy".format(v), length3)
    print(Utility)
    # np.save(dir+save_dir + "\\v_{}_utility.npy".format(v), Utility)
    #
    # np.save(dir + save_dir + "\\v_{}_action1.npy".format(v), action_1)
    # np.save(dir + save_dir + "\\v_{}_action_sum.npy".format(v), action_sum)

    print('')

# print(Para)
#plt.plot(T,Length[0],T,Length[1],T,Length[2],T,Length[3])
#plt.plot(T,Length[0])
# plt.legend(["0","10","50","100"])
# plt.show()


