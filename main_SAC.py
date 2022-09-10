import matplotlib.pyplot as plt
import numpy as np

from applications import *
import Env_SAC as environment
from DRL_SAC import *
import os
def smooth_data(data):
    smooth_data = []
    weight = 0.7
    last = data[0]
    smooth_val = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smooth_val.append(smoothed_val)
        last = smoothed_val
    smooth_data.append(smooth_val)

    return smooth_data[0]


def main():
    applications = (PROC_3D_GAME,VR,AR)
    task_rate = 5
    edge_capability = 60000 * GHZ
    violation_probability = 0.01
    Dicount_Factor = 0.9
    Model = "DRL"
    M = 5
    mmwave_gamma = np.random.gamma(M, 1 / M, 1)
    update_timestep = 1000
    V = [100,150,200,500,1000]
    #V = [500]
    dir = "C:\\Users\\Administrator\\Desktop\\URLLC-Awared Resource Allocation for Heterogeneous Vehicular Edge Computing"
    #dir = "D:\\wwh\\New_nn_two_network_r_s\\New_nn_two_network_r_s\\2022.4.26\\测试保存模型"
    #dir = "/6211924035/wwh/test/New_nn_two_network_r_s/2022.4.26/test_mode_server"
    for v in V:
        time_step = 0
        env = environment.MEC_v1(task_rate, edge_capability, 0, *applications, )
        env.mmwave_gamma = mmwave_gamma[0]
        env.V = v
        actor = DRL_actor(edge_capability, violation_probability, Dicount_Factor, len(applications))
        memory = Memory()
        state = env.init(*applications)
        A_COMM = []
        save_dir  = dir +"\\save_data\SAC_policy\\"+ "V_"+str(v)
        save_mode = dir +"\\save_mode"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(save_mode):
                os.makedirs(save_mode)


        Episode_reward = []
        for episode in range(1,16001):
            state = env.reset()
            length_1 = [];length_2 = [];length_3 = [];
            delay_1 = [] ;delay_2 = [] ;delay_3 = [];
            if episode % 50:
                memory.clear_memory()
            step_reward = 0

            for t in range(1,2001):
                time_step += 1
                #print("-----------------------------------------------------------------------------------")
                print("V "+str(v)+ " episode : "+str(episode)+" , timestep : " + str(t))
                state,r,utility_queue,utility,delay,action_comm,action= env.step(actor, memory,Model)
                step_reward += (-utility_queue)
                memory.rewards.append(-r)
                if time_step % update_timestep == 0:
                    actor.update(memory)
                    time_step = 0
                #print("reward : ",r)
                Length = state.reshape(8, 8)
                #print("Length1 : ",Length[1][5],"Length2 : ",Length[1][6],"Length3 : ",Length[1][7])
                if episode % 1000==0:
                    # 画图
                    length_1.append(Length[1][5])
                    length_2.append(Length[1][6])
                    length_3.append(Length[1][7])
                    delay_1.append(max(delay[6]["mmwave"][0], delay[6]["DSRC"][0], delay[6]["C-V2I"][0]))
                    delay_2.append(max(delay[7]["mmwave"][0], delay[7]["DSRC"][0], delay[7]["C-V2I"][0]))
                    delay_3.append(max(delay[8]["mmwave"][0], delay[8]["DSRC"][0], delay[8]["C-V2I"][0]))
            Episode_reward.append(step_reward/2000)
            if episode % 1000==0:
                x_t = [i for i in range(1, 2001)]
                plt.plot(x_t, length_1, x_t, length_2, x_t, length_3)
                plt.legend(["1", "2", "3"])
                plt.show()
                plt.plot(delay_1)
                plt.title("task1")
                plt.show()
                plt.plot(delay_2)
                plt.title("task2")
                plt.show()
                plt.plot(delay_3)
                plt.title("task3")
                plt.show()
                plt.plot(Episode_reward)
                plt.legend(["reward"])
                plt.show()
                # A_COMM.append(action_comm)
                # bar_width = 0.3
                # Task = ("PROC_3D_GAME","VR","AR")
                # index_mmwave = np.arange(len(Task))+ bar_width  # 男生条形图的横坐标
                # index_DSRC = index_mmwave + bar_width           # 男生条形图的横坐标
                # index_CV2i = index_DSRC  + bar_width            # 男生条形图的横坐标
                # plt.bar(index_mmwave, height= A_COMM[0][0], width=bar_width, color='b', label='mmwave')
                # plt.bar(index_DSRC,   height= A_COMM[0][1], width=bar_width, color='g', label='dsrc')
                # plt.bar(index_CV2i,   height= A_COMM[0][2], width=bar_width, color='orange', label='cv2i')
                # plt.legend(["mmwave","dsrc","cv2i"])
                # plt.show()



        "-----------------------------------------------------保存模型-----------------------------------------------"
        actor.save('SAC_v_{}'.format(v), directory=save_mode)
        # 保存数据
        # np.save(save_dir + "\\delay1.npy", delay_1)
        # np.save(save_dir + "\\delay2.npy", delay_2)
        # np.save(save_dir + "\\delay3.npy", delay_3)
        # np.save(save_dir + "\\length1.npy", length_1)
        # np.save(save_dir + "\\length2.npy", length_2)
        # np.save(save_dir + "\\length3.npy", length_3)
        np.save(save_dir + "\\reward.npy", Episode_reward)
        # np.save(save_dir +"\\Task1_offload_comm_policy", A_COMM[0][0])
        # np.save(save_dir +"\\Task2_offload_comm_policy", A_COMM[0][1])
        # np.save(save_dir +"\\Task3_offload_comm_policy", A_COMM[0][2])





if __name__ == '__main__':
    main()