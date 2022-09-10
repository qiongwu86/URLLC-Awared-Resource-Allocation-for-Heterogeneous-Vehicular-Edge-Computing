import numpy as np
import matplotlib.pyplot as plt
dir = "C:\\Users\\Administrator\\Desktop\\URLLC-Awared Resource Allocation for Heterogeneous Vehicular Edge Computing\\Data\\SAC_policy\\"

V = ["V_0","V_10","V_50","V_100"]
reward_dir = []
reward_data = []
reward_data_ma  = []
reward_data_sd  = []
def moving_average(a, n=40) :
    ret = np.cumsum(a, dtype=float, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def smooth_data(data):
    smooth_data = []
    weight = 0.95
    last = data[0]
    smooth_val = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smooth_val.append(smoothed_val)
        last = smoothed_val
    smooth_data.append(smooth_val)
    return smooth_data[0]

for i in V:
    reward_dir.append(dir+i+"\\reward.npy")
    data = np.load(dir+i+"\\reward.npy")
    reward_data.append(data)
    reward_data_ma.append(moving_average(data))
    reward_data_sd.append(smooth_data(data))

weight = 0.8
#x_r = [i for i in range(1,16001,1)]
#MakerColor = ["r","g","b","m"]
MakerColor = ["darkorange","g","b","m"]
MakerShape = ["*","^","s","o"]
LineShape = ["-","-.","-"]
MarkEvery = 20
V = [0,10,50,100]
for i in range(len(V)):
    plt.subplot(2, 2, i+1)
    #plt.plot(reward_data,alpha = 0.8)
    plt.plot(np.array(reward_data[i][1:])/1e4,MakerColor[i],alpha=0.3)
    plt.plot(np.array(reward_data_sd[i][1:])/1e4, MakerColor[i])
    plt.legend(["V="+str(V[i])])
    plt.xlabel("Episode")
    plt.ylabel("Average reward (1e4)")
plt.subplots_adjust(wspace=0.4,hspace=0.3)
plt.show()

for i in range(len(V)):
    plt.plot(np.array(reward_data[i][1:]) / 1e4, MakerColor[i])

plt.legend(["0","10","50","100"])
plt.show()
# for i in range(len(V)):
#     plt.plot(reward_data_sd[i][1:],MakerColor[i])
#     print(len(reward_data_ma[i]))
#     print(len(reward_data[i]))
#     print(len(reward_data_sd[i][1:]))


