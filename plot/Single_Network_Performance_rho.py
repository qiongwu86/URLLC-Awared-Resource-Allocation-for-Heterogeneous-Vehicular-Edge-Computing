"本代码仿真随机网络演算的延迟上界: 到达速率和延迟 ： 改变车辆"

"dsrc_r = 75 MBPS"
"cv2i_r = 75 MBPS"
"mmwave_b = 20 "

"考虑单个网络的性能(全部分配给某一网络，只有车辆之间的竞争，没有网络之间的竞争，其余任务相同)： "
"   1. 每个任务分配相同的计算资源，dsrc和cv2i的通信带宽一样大，增大到达速率，观察延迟的变化；"
"   2. 每个任务分配相同的计算资源，dsrc和cv2i的通信带宽一样大，增大车辆数目，观察延迟的变化；"

from constants import *
import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
warnings.filterwarnings("ignore")

def net_mode(v2x,gamm):
    if v2x == "DSRC":
        R_dsrc = dsrc_r*MBPS
        t_serv = 0.01
        return  R_dsrc , R_dsrc * t_serv
    elif v2x == "C-V2I":
        R_cv2i = cv2i_r*MBPS
        return R_cv2i
    elif v2x == "mmwave":
        B = mmwave_b*MHZ
        alpha = 2.45
        gamma = 85
        l = 1
        C = np.log2(np.exp(1))*B*np.log(1+gamma*pow(l,-alpha)*gamm)
        #C = cv2i_r*MBPS
        return C

def NetServe(v2x,v2xid,R,B,gamm=None):
    R = R.copy()
    B = B.copy()
    R[TaskId,v2xid] = 0
    B[TaskId,v2xid] = 0    # 这里的仿真是全部由一个通信来传输，所以没有竞争
    if v2x == "DSRC":
        Netmode, latency = net_mode(v2x, gamm=None)
        xi_comm = max(Netmode - sum(R[:, v2x_id[v2x]]),0)
        eta_comm = sum(B[:,v2x_id[v2x]]) + latency # 加一个延迟
        return xi_comm, eta_comm
    elif v2x =="C-V2I":
        xi_comm = net_mode(v2x,gamm=None)
        eta_comm = 0
        return xi_comm,eta_comm
    else:
        Netmode = net_mode(v2x, gamm)
        xi_comm = max(Netmode - sum(R[:, v2x_id[v2x]]),0)
        eta_comm = sum(B[:,v2x_id[v2x]])
        return xi_comm,eta_comm

def CompServe(comp_source,vehicle_n,TaskId,v2xid,R,B,workload,backlog):
    "我这里假设计算资源平分，后面是用action来分配资源"
    R = R.copy()
    B = B.copy()
    R[TaskId,v2xid] = 0
    B[TaskId,v2xid] = 0    # 这里的仿真是全部由一个通信来传输，所以没有竞争
    xi_comp = max(comp_source / workload /vehicle_n - sum(R[TaskId,:]),0)
    print(comp_source / workload /vehicle_n,sum(R[TaskId,:]),sum(B[TaskId,:]),backlog)
    eta_comp = sum(B[TaskId,:])+backlog
    print(xi_comp,eta_comp)
    return xi_comp,eta_comp


def delay(R,B,xi_comp,eta_comp,xi_comm,eta_comm,theta,v_p,TaskId,V2XID):
    rho =  R[TaskId,V2XID]
    sigma =B[TaskId,V2XID]
    "可以拆分成：-np.log(v_p)/(theta*xi_comm)     (sigma + eta_comp + eta_comm)/xi_comm   " \
    " np.log(((np.exp(-theta * xi_comm) - np.exp(-theta * xi_comp)) *(np.exp(theta * xi_comm)   -np.exp( theta * rho)))/(theta*xi_comm)))  "

    "1. 第二项： " \
    "   1.1  当rho比较小的时候 ： xi_comm和xi_comp有值，所以第二项有值； " \
    "   1.2  当rho比较大的时候 ： xi_comm和xi_comp=0， 第二项 = inf， 符合规律当带宽没有资源的时候，延迟无线大"
    "2. 第三项：" \
    "   2.1 当rho比较小的时候 ： np.exp(theta * xi_comm)-np.exp( theta * rho) 大于 0 ， np.log有效，可以计算出来" \
    "   2.2 当rho比较大的时候 ： np.exp(theta * xi_comm)-np.exp( theta * rho) 小于 0 ， np.log无效，出现nan"
    " 也就是竞争剩余的带宽没办法传输改流量：代码改成 max{np.exp(theta * xi_comm)-np.exp( theta * rho),0},然后log里面加1e-20"

    "仿真的结果显示：前两项很小，起决定的时第三项"

    "3. CV2I: 延迟是阶跃型的，不符合规律啊; 第二个问题为什么概率变化没影响" \
    "问题一：原因是 rho*MB > B_cv2i * Mbps ， 这里是符合规律的 ：max{np.exp(theta * xi_comm)-np.exp( theta * rho),0} 就等于0了" \
    "       就是平台为什么上去后有个平台的原因了,平台的原因： np.log(1e-20),但里面是趋向无穷小的，所以结果应该是 inf ，但1e-20导致结果不是inf"
    "       改为 ： log ( ax{np.exp(theta * xi_comm)-np.exp( theta * rho),0} )即可， python会出现 -inf"
    "问题二：原因是第一项代码分母忘记乘以theta"

    "4. mmwave 有一段先升后降再升"
    "   原因： 前面一段comm资源多，所以起决定的是comp；后面一段comp资源多，起决定是comm，等comm资源小于带宽资源，就趋向无穷了"
    "其实我觉得这样解释也无可厚非"

    if rho == 0:
        return 0
    elif xi_comp>=xi_comm:
        First_term = -np.log(v_p)/(xi_comm*theta)
        Second_term = (sigma+eta_comp+eta_comm)/xi_comm
        Third_term = -np.log((np.exp(-theta * xi_comm) - np.exp(-theta * xi_comp)) *\
                    max((np.exp(theta * xi_comm)   -np.exp( theta * rho)),0))/(theta*xi_comm)

        print("xi_comp",First_term,Second_term,Third_term,xi_comp,xi_comm)
        return First_term+Second_term+Third_term
    else:
        First_term = -np.log(v_p)/(xi_comp*theta)
        Second_term = (sigma+eta_comp+eta_comm)/xi_comp
        Third_term = -np.log((np.exp(-theta * xi_comp) - np.exp(-theta * xi_comm)) *\
                    max((np.exp(theta * xi_comp)   -np.exp( theta * rho)),0))/(theta*xi_comp)

        print("xi_comm",First_term, Second_term, Third_term,xi_comp,xi_comm)
        return First_term+Second_term+Third_term

if __name__ == '__main__':
    comp_source = 1500*GHZ
    applications = AR
    #workload = 34532
    workload = 10435
    TaskId = 0
    M = 5
    gamm = np.random.gamma(M, 1 / M, 1)
    V2X = ["mmwave", "DSRC", "C-V2I"]
    v2x_id = {"DSRC": 1,"C-V2I": 2,"mmwave": 0}
    theta = 1e-8
    violation_probability = 0.01
    Latency_vehicle = {}
    #Vehicle_N = [5,10,20]
    Vehicle_N = [5, 10]
    for vehicle in Vehicle_N:
        Latency_vehicle[vehicle] = []
        Latency_network = {"mmwave": [], "DSRC": [], "C-V2I": []}

        xi_comp_network = {"mmwave": [], "DSRC": [], "C-V2I": []}
        xi_comm_network = {"mmwave": [], "DSRC": [], "C-V2I": []}
        for r in range(1,40,1):
            r = r/10
            for v2x in V2X:
                print(v2x)
                propotion = np.zeros(len(V2X))
                propotion[v2x_id[v2x]] = 1
                propotion  = np.array(propotion.tolist()*vehicle).reshape(vehicle,-1)
                R = np.ones((vehicle)) * r * MB
                "PS: 这里burstness是为0"
                B = np.zeros((vehicle))* MB

                R = R.reshape(vehicle,-1)
                B = B.reshape(vehicle,-1)
                # xi_comp = comp_source 是一个常数，因为没有通信之间的竞争
                backlog = 0
                for t in range(1,51):
                    R1 = R * propotion
                    B1 = B * propotion
                    backlog += max(sum(R1[TaskId,:])+sum(B1[TaskId,:])-comp_source / workload /vehicle,0)
                #print(r,v2x,backlog)
                xi_comp,eta_comp = CompServe(comp_source,vehicle,TaskId,v2x_id[v2x],R*propotion, B*propotion, workload,backlog)
                xi_comm,eta_comm = NetServe(v2x, v2x_id[v2x],R*propotion,B*propotion,gamm=gamm)

                xi_comp_network[v2x].append(xi_comp)
                xi_comm_network[v2x].append(xi_comm)

                print("v2x : ",xi_comp,xi_comm)
                d = delay(R*propotion, B*propotion,xi_comp,eta_comp,xi_comm,eta_comm,theta,violation_probability,TaskId,v2x_id[v2x])
                Latency_network[v2x].append(d)

        print("0")

        Latency_vehicle[vehicle].append(Latency_network)


    x_r = [i/10 for i in range(1,40,1)]
    MakerColor = ["r","g","b","m"]
    MakerShape = ["*","^","s","o"]
    LineShape = ["-","-.","-"]
    MarkEvery = 20
    for i in range(len(Vehicle_N)):
        for j in range(len(V2X)):
            n = Vehicle_N[i]
            vehicle_n =  Latency_vehicle[n][0]
            v2x = V2X[j]
            vehicle_d = vehicle_n[v2x]
            MakerSC = MakerShape[i]+MakerColor[j]+LineShape[i]
            plt.plot(x_r,vehicle_d,MakerSC ,markevery=2)

    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 10} # 字体大小，相当于也是leged框的大小
    plt.legend(["mmwave:M=5","DSRC:M=5","C-V2I:M=5",
                "mmwave:M=10","DSRC:M=10","C-V2I:M=10",
                "mmwave:M=10","DSRC:M=10","C-V2I:M=10"],
               loc='upper left', bbox_to_anchor=(0.8, 1),fancybox=True, shadow=False, prop=font2)

    #plt.plot(x_r,Latency["mmwave"],'cx:',x_r,Latency["DSRC"],"rv--",x_r,Latency["C-V2I"],"*g-.",markevery=8)
    #plt.yscale("log")
    plt.xscale("linear")

    plt.ylabel("Delay " + r'$\omega$')
    plt.xlabel("Arrival Rate " + r'$\rho$'+" (Mbps)")

    plt.grid(which="both", linestyle='--')  # 背景网格

    plt.show()






