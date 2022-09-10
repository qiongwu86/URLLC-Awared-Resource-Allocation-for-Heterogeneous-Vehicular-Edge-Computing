import applications
import numpy as np
from constants import *
#from PSO2 import *
import multiprocessing
import os,sys
import pathlib
import warnings
warnings.filterwarnings("ignore")
_parent = str(pathlib.Path(os.getcwd()).parent)
sys.path.append(_parent)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
v2x_mode = ["mmwave", "DSRC", "C-V2I"]
v2x_id = {
    "DSRC":1,
    "C-V2I":2,
    "mmwave":0
}
Task_id = list()
workload = list()
for app_type, info in applications.app_info.items():
    Task_id .append(app_type - 1)
    workload.append(info["workload"])

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
        return C

def NetServe(v2x,v2x_id,rho_net,sigma_net,gamm=None):

    if v2x == "DSRC":
        Netmode, latency = net_mode(v2x, gamm=None)
        xi_comm = max(Netmode - sum(rho_net[:, v2x_id[v2x]]),0)
        eta_comm = sum(sigma_net[:,v2x_id[v2x]]) + latency # 加一个延迟
        return xi_comm, eta_comm
    elif v2x =="C-V2I":
        xi_comm = net_mode(v2x,gamm=None)
        eta_comm = 0
        return xi_comm,eta_comm
    else:
        Netmode = net_mode(v2x, gamm)
        xi_comm = max(Netmode - sum(rho_net[:, v2x_id[v2x]]),0)
        eta_comm = sum(sigma_net[:,v2x_id[v2x]])
        return xi_comm,eta_comm
def CompServe(comp_source,action,task_id,rho_net,sigma_net,workload,backlog_task):

    xi_comp = max(comp_source  * action/ workload - sum(rho_net[task_id, :]),0)
    #xi_comp = max(comp_source*action/ workload - sum(sum(rho_net)), 0)
    #print(xi_comp)
    eta_comp = sum(sigma_net[task_id, :])+backlog_task

    return xi_comp,eta_comp,0

def delay_up(v2x,length,sigma,rho,sigma_net_unchanged,rho_net_unchanged,task_id,v2x_id,epsilon,action_alpha_comp,comp_source,workload,gamm=None):
    rho_net = rho_net_unchanged.copy()
    sigma_net = sigma_net_unchanged.copy()
    rho_net[task_id,v2x_id[v2x]]=0
    sigma_net[task_id,v2x_id[v2x]]=0
    xi_comm, eta_comm = NetServe(v2x=v2x,v2x_id=v2x_id,rho_net=rho_net,sigma_net=sigma_net,gamm=gamm)
    #if v2x == "C-V2I":
        #print(task_id,v2x,length[task_id],rho_net_unchanged[task_id][v2x_id[v2x]])
    xi_comp, eta_comp,_= CompServe(comp_source=comp_source,action=action_alpha_comp[0][task_id],task_id=task_id,rho_net=rho_net,sigma_net=sigma_net,workload = workload,backlog_task=length[task_id])
    theta = 1e-8
    if xi_comp == 0 or xi_comm== 0 :
        print("task : ",task_id ," v2x : ",v2x, " xi_comp : ",xi_comp , " xi_comm : ",xi_comm)

    if rho_net_unchanged[task_id][v2x_id[v2x]] == 0:
        return 0,xi_comp,xi_comm
    elif xi_comp >= xi_comm:
        First_term = -np.log(epsilon) / (xi_comm * theta)
        Second_term = (sigma_net_unchanged[task_id][v2x_id[v2x]] + eta_comp + eta_comm) / (xi_comm)
        Third_term = -np.log((np.exp(-theta * xi_comm) - np.exp(-theta * xi_comp)) * \
                             max((np.exp(theta * xi_comm) - np.exp(theta * rho_net_unchanged[task_id][v2x_id[v2x]])), 0)) / (theta * xi_comm)
        if First_term + Second_term + Third_term>1000:
            print(task_id,v2x,First_term,Second_term,Third_term)
        #if v2x=="C-V2I":
            #print("xi_comp",xi_comm,First_term,Second_term,Third_term)
            #print("xi_comp", sigma_net_unchanged[task_id][v2x_id[v2x]], eta_comp,  eta_comm)
        return min(First_term + Second_term + Third_term,1000),xi_comp,xi_comm
        #return 0, min(First_term + Second_term + Third_term,200000), xi_comp, xi_comm
    else:
        First_term = -np.log(epsilon) / (xi_comp * theta)
        Second_term = (sigma_net_unchanged[task_id][v2x_id[v2x]] + eta_comp + eta_comm) / (xi_comp)
        Third_term = -np.log((np.exp(-theta * xi_comp) - np.exp(-theta * xi_comm)) * \
                             max((np.exp(theta * xi_comp) - np.exp(theta * rho_net_unchanged[task_id][v2x_id[v2x]])), 0)) / (theta * xi_comp)
        #if v2x == "C-V2I":
            #print("xi_comm", xi_comp,First_term, Second_term, Third_term)
        if First_term + Second_term + Third_term>1000:
            print(task_id,v2x,First_term,Second_term,Third_term)
        return min(First_term + Second_term + Third_term,1000),xi_comp,xi_comm
        #return 0, min(First_term + Second_term + Third_term,200000), xi_comp, xi_comm

def rho_sigma_net(action_comm,rho,sigma):
    return action_comm * np.array(rho),action_comm * np.array(sigma)

class Delay:
    def __init__(self,rho,sigma,rho_net, sigma_net,comp_source ,mmwave_gamma,action_comp,fail_epsilon):
        self.rho = rho
        self.sigma = sigma
        self.rho_net =rho_net
        self.sigma_net =sigma_net
        self.comp_source = comp_source
        self.mmwave_gamma =mmwave_gamma
        self.action_comp =action_comp
        self.fail_epsilon = fail_epsilon
        self.v2x_mode = ["mmwave", "DSRC", "C-V2I"]

    def __call__(self, task_id):
        delay ={"mmwave":[], "DSRC":[], "C-V2I":[]}
        for v2x in self.v2x_mode:
            delay[v2x].append(delay_up(v2x=v2x, epsilon=self.fail_epsilon, gamm=self.mmwave_gamma, sigma=self.sigma, rho=self.rho,
                                  sigma_net=self.sigma_net, rho_net=self.rho_net, task_id=task_id, v2x_id=v2x_id,
                                  comp_source=self.comp_source,
                                  action_alpha_comp=self.action_comp, workload=workload[task_id]))
        #return delay
        return task_id,delay

def run_MAS(state,comp_source ,mmwave_gamma,action_comp,action_comm,fail_epsilon,applications):
    # print(applications)
    application = [i-1 for i in applications]
    # print(application)
    # print(state[5])
    # print(state[5][application])
    rho = np.array(state[2][application]).reshape(len(application),1) * MB
    sigma = np.array(state[3][application]).reshape(len(application),1) *MB

    state = state.reshape(8, -1)
    length= state[1][application]

    action_comm = np.array(action_comm)
    rho_net,sigma_net = rho_sigma_net(action_comm,rho,sigma)

    w = {}#
    state_comp = [np.zeros(8) for i in range(3)]
    state_comm = np.zeros(8)
    for task_id in range(len(application)):
        delay = {"mmwave": [], "DSRC": [], "C-V2I": []}

        for v2x in v2x_mode:
            D,State_comp,State_comm = delay_up(v2x=v2x, length=length,epsilon=fail_epsilon, gamm=mmwave_gamma, sigma=sigma, rho=rho,
                                sigma_net_unchanged=sigma_net, rho_net_unchanged=rho_net, task_id=task_id, v2x_id=v2x_id,
                                         comp_source=comp_source,action_alpha_comp=action_comp, workload=workload[application[task_id]])
            delay[v2x].append(D)
            state_comp[v2x_id[v2x]][application[task_id]]=State_comp
            if v2x == "DSRC":
                state_comm[application[task_id]] = State_comm

        w[applications[task_id]]=delay
    state_comp=(np.array(state_comp).reshape(1,-1)/MHZ).tolist()
    state_comm = (np.array(state_comm)/MBPS).tolist()
    State_MAS = np.array(list(state_comm)+list(state_comp[0]))
    return w,State_MAS
