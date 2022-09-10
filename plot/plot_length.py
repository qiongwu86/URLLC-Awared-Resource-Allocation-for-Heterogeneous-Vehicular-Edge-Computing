import numpy as np
import matplotlib.pyplot as plt


def moving_average(a, n=40) :
    ret = np.cumsum(a, dtype=float, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def smooth_data(data):
    smooth_data = []
    weight = 0.999
    last = data[0]
    smooth_val = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smooth_val.append(smoothed_val)
        last = smoothed_val
    smooth_data.append(smooth_val)
    return smooth_data[0]

V = [0,10,50,100]
dir = "C:\\Users\\Administrator\\Desktop\\URLLC-Awared Resource Allocation for Heterogeneous Vehicular Edge Computing"
test_dir  = "\\Data"
random_dir = "C:\\Users\Administrator\\Desktop\\URLLC-Awared Resource Allocation for Heterogeneous Vehicular Edge Computing\\save_data\\Random_policy"
average_dir = "C:\\Users\Administrator\\Desktop\\URLLC-Awared Resource Allocation for Heterogeneous Vehicular Edge Computing\\save_data\\Average_policy"
Length1 = []
Length2 = []
Length3 = []
Action1 =[]
Action_sum = []


for v in V :
    data_dir_1 = dir + test_dir + "\\v_{}_length{}.npy".format(v,1)
    data_dir_2 = dir + test_dir + "\\v_{}_length{}.npy".format(v,2)
    data_dir_3 = dir + test_dir + "\\v_{}_length{}.npy".format(v,3)
    action_dir_1 = dir + test_dir + "\\v_{}_action1.npy".format(v,1)
    action_dir_sum = dir + test_dir + "\\v_{}_action_sum.npy".format(v, 1)
    data1 = np.load(data_dir_1)
    data2 = np.load(data_dir_2)
    data3 = np.load(data_dir_3)
    Length1.append(data1[0::200])
    Length2.append(data2[0::200])
    Length3.append(data3[0::200])
    a1 = smooth_data(np.load(action_dir_1))
    a_sum = smooth_data(np.load(action_dir_sum))
    Action1.append(a1[0::200])
    Action_sum.append(a_sum[0::200])




Random_data_dir_1 =  random_dir + "\\length1.npy"
Random_data_dir_2 =  random_dir + "\\length2.npy"
Random_data_dir_3 =  random_dir + "\\length3.npy"

Random_length_1 = np.load(Random_data_dir_1);
Random_length_2 = np.load(Random_data_dir_2);
Random_length_3 = np.load(Random_data_dir_3);
Length1.append(Random_length_1[0::200])
Length2.append(Random_length_2[0::200])
Length3.append(Random_length_3[0::200])

Average_data_dir_1 =   average_dir + "\\length1.npy"
Average_data_dir_2 =   average_dir + "\\length2.npy"
Average_data_dir_3 =   average_dir + "\\length3.npy"

Average_length_1 = np.load(Average_data_dir_1);
Average_length_2 = np.load(Average_data_dir_2);
Average_length_3 = np.load(Average_data_dir_3);
Length1.append(Average_length_1[0::200])
Length2.append(Average_length_2[0::200])
Length3.append(Average_length_3[0::200])


MakerColor = ["y","g","b","r","m","saddlebrown"]
MakerShape = ["*","^","s","o"]
LineShape = ["-","-.","--",'-']
MarkEvery = 10
font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 10} # 字体大小，相当于也是leged框的大
t2 = [i*200 for i in range(len(Action1[0]))]
for i in range(len(V)):
    # MakerSC = MakerShape[i] + MakerColor[i] + LineShape[i]
    MakerSC = MakerShape[i % 3]+LineShape[i % 3]
    plt.plot(t2,Action1[i],MakerSC,markevery = MarkEvery)

plt.legend(["V=0","V=10","V=50","V=100"])
plt.xlabel("Time steps")
plt.ylabel("Percent of the CPU frequency (%) ")
plt.grid(which="both", linestyle='--')  # 背景网格
plt.show()


MakerColor = ["y","g","b","r","m","saddlebrown"]
MakerShape = ["*","^","s","o"]
LineShape = ["-","-.","--",'-']
MarkEvery = 10
t = [i*200 for i in range(len(Length1[0]))]
font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 10} # 字体大小，相当于也是leged框的大

for i in range(len(V)+2):
    # MakerSC = MakerShape[i] + MakerColor[i] + LineShape[i]

    MakerSC = MakerShape[i % 3] + LineShape[i % 3]
    # MakerSC = MakerColor[i % 3]
    plt.plot(t,Length1[i],MakerSC,markevery = MarkEvery,linewidth=2)
# plt.legend(["V=0","V=10","V=50","V=100","Random","Average"],
#            loc='upper left', bbox_to_anchor=(0.8, 0.55),fancybox=True, shadow=False, prop=font2)

plt.legend(["V=0","V=10","V=50","V=100","RARO","EAEO"],
           loc='upper left', bbox_to_anchor=(0.75, 0.55),fancybox=True, shadow=False, prop=font2)
plt.xlabel("Time steps")
plt.ylabel("Queues backlogs of 3D game tasks")
plt.grid(which="both", linestyle='--')  # 背景网格
plt.show()


for i in range(len(V)+2):
    #MakerSC = MakerShape[i] + MakerColor[i] + LineShape[i]
    MakerSC = MakerShape[i % 3] + LineShape[i % 3]
    # MakerSC = MakerColor[i % 3]
    plt.plot(t, Length2[i], MakerSC, markevery=MarkEvery, linewidth=2)
    # plt.legend(["V=0","V=10","V=50","V=100","Random","Average"],
    #            loc='upper left', bbox_to_anchor=(0.8, 0.55),fancybox=True, shadow=False, prop=font2)

plt.legend(["V=0", "V=10", "V=50", "V=100", "RARO", "EAEO"],
           loc='upper left', bbox_to_anchor=(0.8, 1.01), fancybox=True, shadow=False, prop=font2)
plt.xlabel("Time steps")
plt.ylabel("Queues backlogs of VR tasks")
plt.grid(which="both", linestyle='--')  # 背景网格
plt.show()

for i in range(len(V)+2):
    #MakerSC = MakerShape[i] + MakerColor[i] + LineShape[i]
    print(i)
    MakerSC = MakerShape[i % 3] + LineShape[i % 3]
    # MakerSC = MakerColor[i % 3]
    plt.plot(t, Length3[i], MakerSC, markevery=MarkEvery, linewidth=2)
# plt.legend(["V=0","V=10","V=50","V=100","Random","Average"],
#            loc='upper left', bbox_to_anchor=(0.8, 0.55),fancybox=True, shadow=False, prop=font2)

plt.legend(["V=0", "V=10", "V=50", "V=100", "RARO", "EAEO"],
           loc='upper left', bbox_to_anchor=(0.8, 1.01), fancybox=True, shadow=False, prop=font2)
plt.xlabel("Time steps")
plt.ylabel("Queues backlogs of AR tasks")
plt.grid(which="both", linestyle='--')  # 背景网格
plt.show()


V0_len = []
V10_len = []
V50_len = []
V100_len = []
LineShape = ["--","-","-.","--",'-']
V0_len.append(Length1[0]); V0_len.append(Length2[0]);V0_len.append(Length3[0]);
for i in range(3):
    MakerSC = MakerShape[i % 3] + LineShape[i % 3]
    plt.plot(t,V0_len[i],MakerSC,markevery=MarkEvery)

plt.legend(["3D game tasks", "VR tasks", "AR tasks"],
           loc='upper left', bbox_to_anchor=(0.7, 0.7), fancybox=True, shadow=False, prop=font2)
plt.xlabel("Time steps")
plt.ylabel("Queues backlogs")
plt.grid(which="both", linestyle='--')  # 背景网格
plt.show()


V10_len.append(Length1[1]); V10_len.append(Length2[1]);V10_len.append(Length3[1]);
for i in range(3):
    MakerSC = MakerShape[i % 3] + LineShape[i % 3]
    plt.plot(t,V10_len[i],MakerSC,markevery=MarkEvery)
plt.legend(["3D game tasks", "VR tasks", "AR tasks"],
           loc='upper left', bbox_to_anchor=(0.7, 0.7), fancybox=True, shadow=False, prop=font2)
plt.xlabel("Time steps")
plt.ylabel("Queues backlogs")
plt.grid(which="both", linestyle='--')  # 背景网格
plt.show()
plt.show()

V50_len.append(Length1[2]); V50_len.append(Length2[2]);V50_len.append(Length3[2]);
for i in range(3):
    MakerSC = MakerShape[i % 3] + LineShape[i % 3]
    plt.plot(t,V50_len[i],MakerSC,markevery=MarkEvery)
plt.legend(["3D game tasks", "VR tasks", "AR tasks"],
           loc='upper left', bbox_to_anchor=(0.7, 0.7), fancybox=True, shadow=False, prop=font2)
plt.xlabel("Time steps")
plt.ylabel("Queues backlogs")
plt.grid(which="both", linestyle='--')  # 背景网格
plt.show()
plt.show()

V100_len.append(Length1[3]); V100_len.append(Length2[3]);V100_len.append(Length3[3]);
for i in range(3):
    MakerSC = MakerShape[i % 3] + LineShape[i % 3]
    plt.plot(t,V100_len[i],MakerSC,markevery=MarkEvery)
plt.legend(["3D game tasks", "VR tasks", "AR tasks"],
           loc='upper left', bbox_to_anchor=(0.7, 0.7), fancybox=True, shadow=False, prop=font2)
plt.xlabel("Time steps")
plt.ylabel("Queues backlogs ")
plt.grid(which="both", linestyle='--')  # 背景网格
plt.show()
plt.show()