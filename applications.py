import scipy.stats as stats
import numpy as np
from constants import *

app_info={
    SPEECH_RECOGNITION : {'workload':10435,
        'popularity': 0.5,
        'min_bits':40*KB,
        'max_bits':300*KB
    },NLP : {'workload':25346,
        'popularity': 0.8,
        'min_bits':4*KB,
        'max_bits':100*KB
    },SEARCH_REQ : {'workload':8405,
        'popularity': 10,
        'min_bits':2*BYTE,
        'max_bits':1000*BYTE
    },FACE_RECOGNITION : {'workload':45043,
        'popularity': 0.4,
        'min_bits':10*KB,
        'max_bits':100*KB
    },LANGUAGE_TRANSLATION : {'workload':34252,
        'popularity': 10,
        'min_bits':2*BYTE,
        'max_bits':5000*BYTE
    },PROC_3D_GAME : {'workload':54633,
        'popularity': 0.1,
        'min_bits':0.1*MB,
        'max_bits':3*MB
    },VR : {'workload':40305,
        'popularity': 0.1,
        'min_bits':0.1*MB,
        'max_bits':3*MB
    },AR : {'workload':34532,
        'popularity': 0.1,
        'min_bits':0.1*MB,
        'max_bits':3*MB
    }
}


def app_type_list():
    return app_info.keys()

def app_type_pop():
    '任务流行度'
    # result =[]
    # for i in list(app_info.keys()):

    #     result.append([i, app_info[i]['popularity']])
    # return result
    return [(i, app_info[i]['popularity']) for i in list(app_info.keys())]

def get_info(type, info_name='workload'):
    '返回计算量'
    return app_info[type][info_name]

def arrival_bits(app_type, dist = 'normal', size=1):
    # dpp 那边 只传过来 app_type
    min_bits = app_info[app_type]['min_bits']
    max_bits = app_info[app_type]['max_bits']
    mu = (min_bits+max_bits)/5     #均值
    sigma = (max_bits+min_bits)*6 #方差
    # print("{}\t{}\t{},{},{},{}\t{}".format(app_type, app_info[app_type]['popularity'], mu/MB,sigma/MB,min_bits/MB,max_bits/MB, app_info[app_type]['workload']))
    if dist=='normal':
        if size==1:

            return int(stats.truncnorm.rvs((min_bits-mu)/sigma, (max_bits-mu)/sigma, loc=mu, scale=sigma)),mu,sigma
        return stats.truncnorm.rvs((min_bits-mu)/sigma, (max_bits-mu)/sigma, loc=mu, scale=sigma, size=size).astype(int)
    elif dist=='deterministic':
        return mu
    else:
        return 1

def main():
    result =[]
    for i in range(1,9):   # i= 1-8
        # arrival_bits(i,size=2)) 如果size=2，则[1386766 1870582]
        result.append(app_info[i]['workload']*app_info[i]['popularity']*arrival_bits(i, dist='deterministic'))
        #[7266099200.0, 8637592371.2, 8117829632.0, 34292400.0, 685314016.0, 71035697233.92, 52405941043.200005, 44899688775.68001]
    
    result = np.array(result)/GHZ
    "除以1e9,转换成 Gcycles/s "
    #[7.26609920e+00 8.63759237e+00 8.11782963e+00 3.42924000e-02
    # 6.85314016e-01 7.10356972e+01 5.24059410e+01 4.48996888e+01]
    #import pdb; pdb.set_trace()
    random_task_generation()

def normal_dist(list, mu, sig, peak):
    return peak*np.exp(-(list-mu)**2/2/sig**2)/sig/np.sqrt(2*np.pi)

import matplotlib.pyplot as plt
def random_task_generation():
    import math
    # 24 hours = 86400 secs.
    timeline = np.arange((60*60*24)*7*3) # 1天86400 7天 3个礼拜 [ 0  1  2 ... 1814397 1814398 1814399]
    # weekdays
    mu_days_el = np.array([60*60*7, int(60*60*12.5), 60*60*15, 60*60*19, 60*60*22]) #[25200 45000 54000 68400 79200]
    std_days = [60*60*0.5, 60*60, 60*60, 60*60, 60*60*3]*5
    peak_days = [3,10,1,20,30]*5
    mu_days = np.array([], dtype=int)
    #import pdb; pdb.set_trace()
    for i in range(5):
        mu_days=np.concatenate((mu_days, mu_days_el+i*60*60*24)) #数组拼接
    # weekends
    mu_Sat = np.array([60*60*10, 60*60*15, 60*60*22])+60*60*24*5
    mu_Sun = np.array([60*60*10, 60*60*15, 60*60*21])+60*60*24*6
    std_ends = [60*60*1, 60*60*3, 60*60*6, 60*60*1, 60*60*3, 60*60*2]
    peak_ends = [25,35,45,10,30,20]
    mu = np.concatenate((mu_days, mu_Sat, mu_Sun))
    stds = std_days+std_ends
    peaks = peak_days+peak_ends
    mu = np.concatenate((mu,mu+60*60*24*7,mu+60*60*24*14))
    stds = stds*3
    peaks = peaks*3
    #import pdb; pdb.set_trace()
    graph = np.zeros(60*60*24*7*3)
    for i in range(len(mu)):
        graph += normal_dist(timeline, mu[i], stds[i], peaks[i])
        #print(i)
    #import pdb; pdb.set_trace()
    "应该是任务量乘以任务大小"
    data_size = np.random.poisson(graph*2000)*arrival_bits(2,size=len(timeline))
    #import pdb; pdb.set_trace()
    plt.plot(timeline, data_size)
    plt.show()
    plt.plot(timeline, graph)
    #np.save("graph", np.array([timeline, graph]))
    #np.save("data_size", np.array([timeline, data_size]))
    plt.show()
    # self.arrival_size_buffer.add(arrival_size)
    return graph, data_size
# def sampling_test():
#     for i in range(100):
#         result.append()
if __name__=='__main__':
    main()
    # random_task_generation()
