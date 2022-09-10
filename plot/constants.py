
# Channels
'信道'
LTE = 1
WIFI = 2
BT = 3 #蓝牙
NFC = 4
WIRED = 5

'应用'
# Applications
SPEECH_RECOGNITION = 1
NLP = 2
SEARCH_REQ = 3
FACE_RECOGNITION = 4
LANGUAGE_TRANSLATION = 5
PROC_3D_GAME = 6
VR = 7
AR = 8

'数据大小'
# Data size scales
BYTE = 8    #8位
KB = 1024*BYTE
MB = 1024*KB
GB = 1024*MB
TB = 1024*GB
PB = 1024*TB

# CPU clock frequency scales
KHZ = 1e3
MHZ = KHZ*1e3
GHZ = MHZ*1e3

# Data transmission rate scales
'数据传输'
KBPS = 1e3
MBPS = KBPS*1e3
GBPS = MBPS*1e3

# Time scales
MS = 1e-3

dsrc_r = 27
cv2i_r = 27
mmwave_b = 20

'''
arrival rate            Mbps
arrival data size       Mbps
time slot interval      sec (TBD)
Edge computation cap.   3.3*10^2~10^4
'''

def main():
    import numpy as np
    # result =[]
    # for i in range(1,9):
    #     result.append(app_info[i]['workload']*app_info[i]['popularity']*arrival_bits(i, dist='deterministic'))
    # result = np.array(result)/GHZ
    import pdb; pdb.set_trace()
    #程序运行到这里就会暂停
    #用于调试代码的常用库

if __name__=='__main__':
    main()
