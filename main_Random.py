import matplotlib.pyplot as plt
import numpy as np

from applications import *
import Env as environment
from DRL_SAC import *
dir = "C:\Users\Administrator\Desktop\URLLC-Awared Resource Allocation for Heterogeneous Vehicular Edge Computing\save_data\Random_policy"
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
    Model = "Random"
    env = environment.MEC_v1(task_rate, edge_capability, *applications)
    M = 5
    mmwave_gamma = np.random.gamma(M, 1 / M, 1)
    env.mmwave_gamma = mmwave_gamma[0]
    actor = DRL_actor(edge_capability, violation_probability , Dicount_Factor, len(applications))
    memory = Memory()
    state = env.init(*applications)
    delay_1  = [] ; delay_2  = [] ; delay_3  = [] ;
    length_1 = [] ; length_2 = [] ; length_3 = [] ;
    Utility = []
    for t in range(1,16001): # max_timesteps = 2000
        print(t)
        state,_, delay,utility,_,_,_ = env.step(actor, memory,Model)
        state = state.reshape(8, -1)
        length_state = state[1]

        delay_1.append(delay[5])
        delay_2.append(delay[6])
        delay_3.append(delay[7])
        length_1.append(length_state[5])
        length_2.append(length_state[6])
        length_3.append(length_state[7])
        Utility.append(utility)


    #delay_1_smooth = smooth_data(delay_1)
    # np.save(dir+"delay1.npy",delay_1)
    # np.save(dir+"delay2.npy",delay_2)
    # np.save(dir+"delay3.npy",delay_3)
    # np.save(dir + "length1.npy", length_1)
    # np.save(dir + "length2.npy", length_2)
    # np.save(dir + "length3.npy", length_3)
    # np.save(dir + "utility.npy", Utility)
    plt.plot(delay_1,alpha=0.75);
    #plt.plot(delay_1_smooth);
    plt.show()
    plt.plot(delay_2);
    plt.show()
    plt.plot(delay_3)
    plt.show()
    print("------------------------------------------------------")
    # print(length_1)
    # print(length_2)
    # print(length_3)
    plt.plot(length_1);plt.plot(length_2);plt.plot(length_3)
    plt.legend(["1","2","3"])
    plt.show()
    print("stop")

if __name__ == '__main__':
    main()