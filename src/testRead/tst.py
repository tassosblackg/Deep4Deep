#test reading .cmp files
import imp
imp.load_source("read_img","/home/tassos/Desktop/SpeechR2/Deep4Deep/src/read_img.py")
import numpy as np
import matplotlib.pyplot as plt
import read_img as rim

#rim.read_cmp_file("testRead/T_1000001.cmp")

#check read dir_name
# cmpl=rim.read_cmp_dir("ASVspoof2017_V2_train_fbank")
# print(cmpl[0],cmpl[1])
#check read labels--ok -chk onnly path of protocol_V2 dir
#cl_types= rim.read_label("train_info.txt")
#print(cl_types)

#check read data
Xdata,Υdata,nframes=rim.read_Data("ASVspoof2017_V2_train_fbank","train_info.txt",80)
print(nframes)
