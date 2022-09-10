import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.pyplot import MultipleLocator
#import pydevd
#pydevd.settrace('211.65.24.183', port=51046, stdoutToServer=True, stderrToServer=True)

# V = [0,10,50,100]
V = [0,10,50,100]
dir = "C:\\Users\\Administrator\\Desktop\\URLLC-Awared Resource Allocation for Heterogeneous Vehicular Edge Computing"
save_dir  = "\\Data"
random_dir = "C:\\Users\Administrator\\Desktop\\URLLC-Awared Resource Allocation for Heterogeneous Vehicular Edge Computing\\save_data\\Random_policy"
average_dir = "C:\\Users\Administrator\\Desktop\\URLLC-Awared Resource Allocation for Heterogeneous Vehicular Edge Computing\\save_data\\Average_policy"

def total_num(data):
    data_dict = {}
    for key in data:
        data_dict[key] = data_dict.get(key, 0) + 1
    return data_dict

def grand_total(data):
    # 计算互补累积概率分布 num_days中存储的是所有的次数，总次数为sum(num_days)
    num_data = total_num(data)
    sorted_data={}
    for i in sorted(num_data):
        sorted_data[i]=num_data[i]
    Data_v = []
    Data_num =[]
    for k,v in sorted_data.items():
        if k ==10000:
            continue
        Data_v.append(k)
        Data_num.append(v)
    grand_total = []
    for i in range(1, len(Data_num)):
        a = Data_num[i:]  # 互补累积概率，单调递减
        p = sum(a) / sum(Data_num)
        grand_total.append(p)
    # print(grand_total)
    grand_total.append(0.0)
    return Data_v,grand_total
Latency_1_ccdf_v = [] ; Latency_1_ccdf_p = []
Latency_2_ccdf_v = [] ; Latency_2_ccdf_p = []
Latency_3_ccdf_v = [] ; Latency_3_ccdf_p = []
for v in V :
    data_dir_1 = dir + save_dir + "\\v_{}_delay{}.npy".format(v,1)
    data_dir_2 = dir + save_dir + "\\v_{}_delay{}.npy".format(v,2)
    data_dir_3 = dir + save_dir + "\\v_{}_delay{}.npy".format(v,3)

    delay_1 = np.load(data_dir_1)
    delay_2 = np.load(data_dir_2)
    delay_3 = np.load(data_dir_3)

    v1,p1 = grand_total(delay_1) ; Latency_1_ccdf_v.append(v1) ; Latency_1_ccdf_p.append(p1) ;
    v2,p2 = grand_total(delay_2) ; Latency_2_ccdf_v.append(v2) ; Latency_2_ccdf_p.append(p2) ;
    v3,p3 = grand_total(delay_3) ; Latency_3_ccdf_v.append(v3) ; Latency_3_ccdf_p.append(p3) ;

Random_data_dir_1 =  random_dir + "\\delay1.npy"
Random_data_dir_2 =  random_dir + "\\delay2.npy"
Random_data_dir_3 =  random_dir + "\\delay3.npy"

Random_delay_1 = np.load(Random_data_dir_1)
Random_delay_2 = np.load(Random_data_dir_2)
Random_delay_3 = np.load(Random_data_dir_3)

v1,p1 = grand_total(Random_delay_1) ; Latency_1_ccdf_v.append(v1) ; Latency_1_ccdf_p.append(p1) ;
v2,p2 = grand_total(Random_delay_2) ; Latency_2_ccdf_v.append(v2) ; Latency_2_ccdf_p.append(p2) ;
v3,p3 = grand_total(Random_delay_3) ; Latency_3_ccdf_v.append(v3) ; Latency_3_ccdf_p.append(p3) ;


Average_data_dir_1 =  average_dir + "\\delay1.npy"
Average_data_dir_2 =  average_dir + "\\delay2.npy"
Average_data_dir_3 =  average_dir + "\\delay3.npy"

Average_delay_1 = np.load(Average_data_dir_1)
Average_delay_2 = np.load(Average_data_dir_2)
Average_delay_3 = np.load(Average_data_dir_3)

v1,p1 = grand_total(Average_delay_1) ; Latency_1_ccdf_v.append(v1) ; Latency_1_ccdf_p.append(p1) ;
v2,p2 = grand_total(Average_delay_2) ; Latency_2_ccdf_v.append(v2) ; Latency_2_ccdf_p.append(p2) ;
v3,p3 = grand_total(Average_delay_3) ; Latency_3_ccdf_v.append(v3) ; Latency_3_ccdf_p.append(p3) ;

MakerColor = ["y","g","b","m","r","saddlebrown"]
MakerShape = ["*","^","s","o"]
LineShape = ["-","-","-",'-']
MarkEvery = 150
for i in range(len(V)+2):
    # MakerSC = MakerShape[i] + MakerColor[i] + LineShape[i]
    MakerSC = MakerColor[i]
    plt.plot(Latency_1_ccdf_v[i],Latency_1_ccdf_p[i], MakerSC,markevery=MarkEvery )
plt.legend(["V=0","V=10","V=50","V=100","RARO","EAEO"])

# x_major_locator= MultipleLocator(50)
ax=plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)
plt.yscale("log")
plt.xscale("log")
ax.set_xticks([30,50, 300, 500,1000])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.ylabel("Violation Probability  " + r'$\epsilon_{1}$')
plt.xlabel("Delay upper bound " + r'$\omega_{1}$' + " (ms) ")
plt.grid(which="both", linestyle='--')  # 背景网格
plt.show()

for i in range(len(V)+2):
    #MakerSC = MakerShape[i] + MakerColor[i] + LineShape[i]
    MakerSC = MakerColor[i]
    plt.plot(Latency_2_ccdf_v[i],Latency_2_ccdf_p[i], MakerSC,markevery=MarkEvery )
plt.legend(["V=0","V=10","V=50","V=100","RARO","EAEO"])
ax=plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)

plt.yscale("log")
plt.xscale("log")
ax.set_xticks([30,50, 300, 500,1000])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.ylabel("Violation Probability  " + r'$\epsilon_{2}$')
plt.xlabel("Delay upper bound " + r'$\omega_{2}$' + " (ms) ")
plt.grid(which="both", linestyle='--')  # 背景网格
plt.show()

for i in range(len(V)+2):
    #MakerSC = MakerShape[i] + MakerColor[i] + LineShape[i]
    MakerSC = MakerColor[i]
    plt.plot(Latency_3_ccdf_v[i],Latency_3_ccdf_p[i], MakerSC,markevery=MarkEvery )
plt.legend(["V=0","V=10","V=50","V=100","RARO","EAEO"])
plt.yscale("log")
plt.xscale("log")
ax=plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)
ax.set_xticks([30,50, 300, 500,1000])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.ylabel("Violation Probability  " + r'$\epsilon_{3}$')
plt.xlabel("Delay upper bound " + r'$\omega_{3}$' + " (ms) ")
plt.grid(which="both", linestyle='--')  # 背景网格
plt.show()

