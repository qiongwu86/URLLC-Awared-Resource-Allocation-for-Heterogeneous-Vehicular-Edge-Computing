
from constants import *
import numpy as np
import collections
import applications
from cost_functions import *
import torch
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#from MAS_main2_appro_delay import *
from SNC_RL import *
v2x_mode = ["mmwave", "DSRC", "C-V2I"]
v2x_cost = {"mmwave":10,"DSRC":5,"C-V2I":2}
class Task(object):
    def __init__(self, app_type, data_size,  arrival_timestamp=None):
        " data_size = np.random.poisson(task_rate*population)*applications.arrival_bits(app_type) "
        self.app_type = app_type
        self.data_size = data_size
        self.arrival_timestamp = arrival_timestamp

class TaskBuffer:
    "t时刻的数据长度"
    def __init__(self, max_size=100):#, initial_storage=0): max_size 可以存放数据长度信息的buffer
        # super.__init__(max_size)
        self.storage = list()
        self.max_size = max_size

    def add(self, data):
        self.storage.append(data) # [ (0,7241504) ],length=1
        if len(self.storage)>self.max_size:
            self.storage =self.storage[1:]  # 第一个是越久以前的数据长度,不要了
        else:
            pass

    def get_buffer(self):
        #倒序返回
        # [(1,100,rho,sigma),(2,200,rho,sigma)] --> [(2, 200,rho,sigma), (1, 100,rho,sigma)] 元组 arrival_timestamp, task_length 越后面是越新的
        return self.storage[::-1]

    def get_last_obj(self):
        if self.storage:
            return self.storage[-1] # 最后一个是上一个时刻的数据长度信息
        else:
            #创建时storage为空
            return None

class TaskQueue(object):
    def __init__(self, app_type=None, max_length=np.inf):
        self.max_length = max_length  # 队列无穷大
        self.tasks = collections.OrderedDict()
        self.app_type = app_type
        self.arrival_size_buffer = TaskBuffer(max_size=100)  # 一个Task对应一个Buffer
        self.length = 0  # 目前记录的数据长度
        self.exploded = 0

    def mean_arrival(self, t, interval=10, scale=1):
        #print("env.timestep : ",t, " buffer数据 : ",self.arrival_size_buffer.get_buffer())
        result = 0
        for time, data_size, _ ,_ in self.arrival_size_buffer.get_buffer():
            # 这个代码写的很巧妙，不用分别处理buffer里面是不是有100个数据的情况
            # time = 44，43，42，41，40，39 。。。。。。。。。。
            if time > t - interval:  # 算100个时刻以内的数据长度
                result += data_size  #
            else:                    #
                break  # result = 0
        # 因为t是从0开始的
        return result/min(t+1,interval)/scale  # 计算这段时间间隔类的平局长度()

    def last_arrival(self, t, scale=1):
        "self.arrival_size_buffer 是 taskbuffer类 "
        last_data = self.arrival_size_buffer.get_last_obj() # 返回上一个时刻的长度
        if last_data: #创建时返回为None
            time, data_size,rho,sigma =last_data
            if time==t:  # 该时刻t是队列里的time时刻
                return data_size/scale,rho,sigma
        return 0,0,0 #创建时返回为None

    def arrived(self, task_length, task,rho,sigma,arrival_timestamp):  # task, arrival_timestamp
        self.arrival_size_buffer.add((arrival_timestamp, task_length,rho,sigma))  # 注意放进去的是元组
        #print("env.timestep : ", arrival_timestamp, " buffer数据 : ", self.arrival_size_buffer.get_buffer())
        if self.length + task_length <= self.max_length: # get_length()是当前队列的长度  task_length是任务长度 ，max_length = inf
            "self.max_length 考虑以后队列有限长的情况"
            self.tasks[arrival_timestamp] = task
            self.length += task_length
            self.exploded = max(0, self.exploded-1) # ？？
            return True
        else:
            "超出就删除该任务"
            self.exploded = min(10, self.exploded+1) # ？？
            return False

    def served(self, resource): # my_task_queue.served(cpu_allocs[app_type]*self.computational_capability, type=1)
        "resource = cpu_allocs[app_type]*self.computational_capability"
        if resource == 0: # 需要用的CPU资源 = 0 resource = cpu_allocs[app_type]*self.computational_capability
            return
        else:
            task_to_remove = []
            served = 0
            to_be_served = int(resource/applications.get_info(self.app_type,'workload'))
            for task_id, task_ob in self.tasks.items():
                "  TaskQueue 有新的也有旧的[(1,2),(3,4)...] "
                task_size = task_ob.data_size  # bits
                if to_be_served >= task_size: # 分配的CPU资源 > 任务大小
                    task_to_remove.append(task_id)  #to_be_served >= task_size，每一次step都会创建task
                    "这个是留给同一个queue后面的task处理的，也就是第一个task处理完，剩下的处理第二个"
                    to_be_served -= task_size  #
                    served += task_size #
                elif to_be_served > 0:  # 没处理完
                    "任务需要的计算资源(cycles) >  计算资源(cycles)"
                    task_size -= to_be_served # 还没服务完的任务 or 还灭卸载完的任务
                    self.tasks[task_id].data_size = task_size
                    self.length -= to_be_served # 这里只是减去cpu处理完的任务量，并没有减去卸载到云端的量
                    served += to_be_served
                    # if not silence: print("remained queue_length of type{} : {}".format(self.app_type, self.length))
                    to_be_served = 0
                    task_useall = task_id
                else:  # to_be_served = 0 "处理前面task的cpu都已经用完"
                    #print("处理%d任务时CPU资源全部用完,后面任务没有资源可用"% task_useall)
                    break
            resource = served * applications.get_info(self.app_type,'workload') # 用过的资源(cycles)
            self.remove_multiple_tasks(task_to_remove)
            self.print_queue_task()
            return resource

    def remove_multiple_tasks(self, task_list):
        for task_id in task_list:
            self.length -= self.tasks[task_id].data_size
            del self.tasks[task_id]

    def print_queue_task(self):
        task_datasize = list()
        for _, task in self.tasks.items():
            task_datasize.append(task.data_size)
        #print("队列里的任务 : ", list(zip(self.tasks.keys(), task_datasize)))


class MasState(object):
    def __init__(self, app_type=None, max_length=np.inf):
        self.Mas_State = np.zeros(4*8)

    def get_MasState(self):
        return self.Mas_State

class MEC_v1():
    def __init__(self, task_rate, edge_capability,*applications, time_delta=10*MS,empty_reward=True, fail_epsilon=0.01):
        super().__init__()
        self.applications = applications
        self.task_rate = task_rate# = 10
        self.empty_reward = empty_reward  #True


        self.queue_list = {}
        self.timestamp = 0

        self.computational_capability = edge_capability
        self.mmwave_gamma = 0
        self.fail_epsilon = fail_epsilon
        self.before_arrival  = [0,0,0,0,0,0,0,0]

    def init(self,*application_types):
        "每一次episode后reset就删了self.queue_list"
        for application_type in application_types:
            self.queue_list[application_type] = TaskQueue(application_type)
        self.MasState = MasState()
        state,_ = self.get_status(MasState=None)
        self.state_dim = len(state)                 # 64
        self.action_dim = len(self.applications) + 1  # 8+1 = 9
        return state

    def __del__(self):
        del self.queue_list
        del self.MasState

    def reset(self, empty_reward=True):
        task_rate = self.task_rate # 10
        edge_capability = self.computational_capability
        applications = self.applications #(1,2,3)
        mmwave_gamma = self.mmwave_gamma
        self.__del__()
        self.__init__(task_rate, edge_capability,*applications)
        self.mmwave_gamma = mmwave_gamma
        reset_state = self.init(*applications)
        qlength = self.get_total_qlength()
        print("env.timestep : ",self.timestamp,"队列总长度 : ",qlength)
        return reset_state

    def step(self,actor,memory,Model):

        q0, failed_to_generate,q1,state1= self._task_generation()   # q0 是没产生任务之前的队列长度，q1 是产生任务之后的队列长度
        if Model=="DRL":
            action,action_comm= actor.choose_action(state1,memory) # ndarray : [[a,b,c,d]] , [[a,b,c],[a,b,c],[a,b,c]]
        elif Model=="Average":
            action =np.ones([1,4])*(1/3)
            action_comm = np.ones([3,3])*(1/3)
        elif Model=="Random":
            # a = np.random.rand(1,4)
            # action = a/sum(sum(a))
            # a_c = np.random.rand(3,3)
            # action_comm = a_c/np.sum(a_c,1).reshape(-1,1)
            a = np.random.rand(1,4)
            action = a/sum(sum(a))
            action_comm = np.ones([3, 3]) * (1 / 3)

        used_edge_cpus, state2, q2 = self._step_alpha(action)       # q2是 采取动作之后的队列长度
        self.before_arrival = q2
        bt = np.array(q1)-np.array(q2)
        delay,MAS_State= run_MAS(state1.reshape(8, -1),self.computational_capability, self.mmwave_gamma,
                         action, action_comm, self.fail_epsilon,self.applications)

        self.MasState.Mas_State = MAS_State[-4*8:].tolist().copy()
        # next_state,_= self.get_status(MasState=self.MasState.get_MasState())
        "添加新的状态input:(memory.next_state,next_state)"
        "----------------不用添加吧，因为是用state1的状态算reward-------------------------------------------------"
        latency,utility,plot_cost= self.get_cost(state1,action,action_comm,used_edge_cpus,delay,q0,bt)
        self.timestamp +=1
        return state2, action ,latency ,utility,plot_cost,used_edge_cpus,delay
    def _task_generation(self):
        Rho = list();Sigma = list()
        initial_qlength = self.get_total_qlength()
        app_type_pop = applications.app_type_pop()  # 流行度  [(1, 0.5), (2, 0.8), (3, 0.4), (4, 10), (5, 1), (6, 0.1), (7, 0.1), (8, 0.1)]
        this_app_type_list = list(self.queue_list.keys())  # [1, 2, 3]
        arrival_size = np.zeros(8) # 当前时刻产生数据的大小
        failed_to_generate = 0  # 创建失败的个数
        for app_type, population in app_type_pop:
            if app_type in this_app_type_list:
                data_size,rho,sigma= applications.arrival_bits(app_type)
                task_num = np.random.poisson(self.task_rate * population)+1
                # print(app_type,task_num)
                rho*=task_num; sigma*=task_num;
                Rho.append([rho]);Sigma.append([sigma])
                data_size = rho+sigma
                task = Task(app_type, data_size, arrival_timestamp=self.timestamp)
                failed_to_generate += (not self.queue_list[app_type].arrived(data_size,task,rho,sigma,self.timestamp))
                arrival_size[app_type - 1] = data_size
        "打印队列里的任务"
        for app_type, population in app_type_pop:
            if app_type in this_app_type_list:
                self.queue_list[app_type].print_queue_task()
        after_qlength = self.get_total_qlength()
        state, _, = self.get_status(MasState=self.MasState.get_MasState())
        #print("产生任务队列总长度 : ",after_qlength)
        return initial_qlength,failed_to_generate,after_qlength,state

    def _step_alpha(self, action):  # [[0.262274   0.20443535 0.3213151  0.2119755 ]]
        used_edge_cpus = collections.defaultdict(float) ##使用collections类中的defaultdict()方法来为字典提供默认值
        action = action.flatten()[:len(self.applications)].reshape(1,-1)[0]
        print(action)
        app_type_list = list(self.queue_list.keys())  # [1,2,3,4,5,6,7,8]
        cpu_allocs = dict(zip(app_type_list, action))  # 打包成一对 然后变成字典
        for app_type in app_type_list:
            if cpu_allocs[app_type] == 0 or (app_type not in self.queue_list.keys()): # 如果app_type的action=0，即需要使用的cpu为0
                pass
            else:
                my_task_queue = self.queue_list[app_type]  # self.queue_list ={1:TaskQueue类,2:TaskQueue类,3:TaskQueue类}
                if my_task_queue.length:
                    cpu_allocs[app_type]= my_task_queue.served(cpu_allocs[app_type] * self.computational_capability)
                else:  #队列总长度为0，则使用CPU为0
                    cpu_allocs[app_type] = 0

        state, _,= self.get_status(MasState=self.MasState.get_MasState())
        after_qlength = self.get_total_qlength()
        #print("处理任务队列总长度 : ", after_qlength)
        return sum(cpu_allocs.values()),state, after_qlength

    def get_total_qlength(self):
        qlength = list([0,0,0,0,0,0,0,0])
        for i, queue in self.queue_list.items():  # return self.queue_list.items() ： _ = uuid , queue = TaskQueue
            qlength[i-1]=queue.length
        return qlength

    def get_status(self,MasState):
        edge_state= list()
        failed_to_generate = 0
        #queue_estimated_arrivals = np.zeros(8)  # 平均长度

        queue_arrivals = np.zeros(8)            # 上一时刻
        queue_lengths = np.zeros(8)

        #queue_exploded = np.zeros(8)

        queue_rho = np.zeros(8)
        queue_sigma = np.zeros(8)
        #app_info = np.zeros(8)
        if MasState != None:
            left_dsrc = MasState[:8]
            left_mmwave_comp = MasState[8:16]
            left_dsrc_comp = MasState[16:24]
            left_v2i_comp = MasState[24:32]
        else:
            left_dsrc = np.zeros(8)
            left_mmwave_comp = np.zeros(8)
            left_dsrc_comp = np.zeros(8)
            left_v2i_comp = np.zeros(8)
        for app_type, queue in self.queue_list.items(): # self.queue_list：{"1": TaskQueue, "2":TaskQueue,"3":TaskQueue }
            _ = queue.mean_arrival(self.timestamp, interval=100) # 计算这段时间间隔(time, estimate_interval)内的平均长度
            queue_arrivals[app_type-1],queue_rho[app_type-1],queue_sigma[app_type-1]= queue.last_arrival(self.timestamp) # 上一个时刻的长度
            queue_lengths[app_type-1] = queue.length
            if MasState == None:
                left_dsrc[app_type-1] =  dsrc_r * MBPS / MBPS
                left_mmwave_comp[app_type-1] = self.computational_capability/GHZ
                left_dsrc_comp[app_type-1] = self.computational_capability/GHZ
                left_v2i_comp[app_type-1] = self.computational_capability/GHZ

        temp_state = list(np.array(queue_arrivals)/MB)+list(np.array(queue_lengths)/MB)\
                     +list((np.array(queue_rho)/ MB).tolist())+list((np.array(queue_sigma)/MB).tolist())\
                     +list(left_dsrc)+list(left_mmwave_comp)+list(left_dsrc_comp)+list(left_v2i_comp)
        failed_to_generate = 0
        self.MasState.Mas_State = temp_state[-4*8:].copy()
        return np.array(temp_state),failed_to_generate

    def get_cost(self,state,action,action_comm,used_edge_cpus,delay,q0,bt): #used_edge_cpus, used_cloud_cpus, q0, q3, failed_to_offload, failed_to_generate
        def compute_cost_fct(cores, cpu_usage):         # core = 10
            return cores*(cpu_usage/400/GHZ/cores)**3*1000
        this_app_type_list = list(self.queue_list.keys())
        state1 = state.reshape(8, -1)  # 转换成5行
        arrivals = state1[0] * MB   # 当前任务的大小
        CV2I_COST = 0
        pay_cost = 100
        for i in range(len(this_app_type_list)):
            CV2I_COST += action_comm[i][2]*state1[0][this_app_type_list[i]-1]*500
        action = action[0][:3]
        # Task_require = [0.25,2,2,0.25,2,2,2,0.4]
        Task_require = [0.3,0.15, 0.08, 100, 100, 100, 100, 100]
        Exceed_delay = 0
        qt = np.array(q0)
        Edge_drift_cost = sum((qt/MB)*((arrivals-bt)/MB))+1/2*sum(((arrivals-bt)/MB)**2)
        Computation_cost = compute_cost_fct(10, self.computational_capability*sum(action))/1e3
        Cost = CV2I_COST + Computation_cost
        Latency = np.zeros(8)
        for i in this_app_type_list:
            Latency[i-1] = max(delay[i]["mmwave"][0],delay[i]["DSRC"][0],delay[i]["C-V2I"][0])
        #reward = Edge_drift_cost+5*np.mean(Latency)

        return Latency,Cost,0