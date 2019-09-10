# script to read and calculate the train data mean/std
import read_img as rm
import numpy as np
import os
path_to_train_set = "/home/tassos/Desktop/DATA_ASR/ASVspoof2017_V2_train_fbank"
total_inp_files = len(os.listdir(path_to_train_set))
# calcualte mean and std on train s_utterance
def calculate_statistics_save(mean,std):

    string = str(mean)+','+str(std)
    f = open('statistics','w')
    f.write(string)
    f.close()


mean_x = []
std_x = []
for i in range(1,total_inp_files,300):
    Xt,Yt,size = rm.read_Data("ASVspoof2017_V2_train_fbank", "train_info.txt",300)
    mean_x.append(np.mean(Xt))
    std_x.append(np.std(Xt))

total_mean_X = np.sum(mean_x)/len(mean_x)
# print(total_mean_X)
# print(np.power(std_x,2))
total_std_X = np.sqrt(np.sum(np.power(std_x,2))/len(std_x))
calculate_statistics_save(total_mean_X,total_std_X)
