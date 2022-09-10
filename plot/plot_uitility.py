import numpy as np
import matplotlib.pyplot as plt
def smooth_data(data):
    smooth_data = []
    weight = 0.8
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
random_dir = "\\save_data\\Random_policy"
average_dir = "\\save_data\\Average_policy"
Utility = []
for v in V :
    data_dir_1 = dir + test_dir + "\\v_{}_utility.npy".format(v)
    u = smooth_data(np.load(data_dir_1))
    Utility.append(u[0::50])
Random_data_dir = dir + random_dir + "\\utility.npy"
Random_utility = smooth_data(np.load(Random_data_dir))
Utility.append(Random_utility[0::50])

Average_data_dir = dir + average_dir + "\\utility.npy"
Average_utility = smooth_data(np.load(Average_data_dir))
Utility.append(Average_utility[0::50])



MakerColor = ["y","g","b","r","m","saddlebrown","c"]
MakerShape = ["*","^","s","o","<",">","+"]
LineShape = ["-","-","-",'-',"-","-","-"]
MarkEvery = 150
t = [i*50 for i in range(len(Utility[0][1:]))]
font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 8} # 字体大小，相当于也是leged框的大小
for i in range(len(V)+2):
    # MakerSC = MakerShape[i] + MakerColor[i] + LineShape[i]
    print(i)
    MakerSC = MakerColor[i]
    plt.plot(t,Utility[i][1:], MakerSC)
# plt.legend(["V=0","V=10","V=50","V=100","Random","Average"],
#            loc='upper left', bbox_to_anchor=(0.8, 0.9),fancybox=True, shadow=False, prop=font2)
plt.legend(["V=0","V=10","V=50","V=100","RARO","EAEO"],
           loc='upper left', bbox_to_anchor=(0.8, 0.8),fancybox=True, shadow=False, prop=font2)
plt.xlabel("Time steps")
plt.ylabel("System utility " + r'$F(t)$')
plt.grid(which="both", linestyle='--')  # 背景网格
plt.show()

