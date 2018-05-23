# script for reading each .cmp file from directories
# and then create images
import os
import glob as g
import numpy as np
import matplotlib.pyplot as plt


path="../protocol_V2" # check and define it properly

# read if .wav file is genuine or spoof --create label
# as filename put the full path of train_info file in protocolv2 dir
def read_label(filename):
    full_path=os.path.join(path,filename)
    #files_n = [] #save .wav file's name
    class_type = [] #save labels
    with open(full_path, 'r') as f:
        # read line by line
        for line in f:
            l_words = line.split(" ")  # seperate words by space
            #files_n.append(l_words[0])
            if(l_words[1] == "genuine"):
                class_type.append('1')
            elif (l_words[1] == "spoof"):
                class_type.append('0')
            else:
                pass

    return class_type


# read  all files' name from directory and create an array of them
def read_dir(dir_name):
    files = g.glob(dir_name+"/*.cmp")
    cmp_list = []
    for i in range(files.__len__()):
        cmp_list.append(read_cmp_file(files[i]))
    return cmp_list


# read .cmp a file and converted to numpy array (transposed)
def read_cmp_file(cmp_filename):
    nfilt = 64
    with open(cmp_filename, 'rb') as fid:
        cmp_data = np.fromfile(fid, dtype=np.float32, count=-1)

    cmp_data = cmp_data.reshape((-1, nfilt))  # where nfilt = 64
    # @ check if read is done right
    # fbank = 10  # put a number between 0 and 63
    #plt.plot(cmp_data[: fbank])
    # plt.show()
    return (cmp_data)

#read all input image files of a directory
#and labels from each files
#args:
# @dir_name : directory name where .cmp files are saved
# @info_fl : text file where info about train,eval,dev sets are saved
def read_Data(dir_name,info_fl):
    cmp_nl=read_dir(dir_name) #read .cmp files from dir
    cl_types=read_label(info_fl) # read label from info file
    data_l=[]
    #for each cmp file
    for i in range (cmp_nl.__len__()):
        cmp_data=read_cmp_file(cmp_nl[i]) #read that file
        cmp2img=convert_to_images(cmp_data) #convert this file to image
        data_l.append(cmp2img,cl_types[i]) # image object appent to a list
    return data_l

# ------------------------------------------------------------------------------
# convert .cmp file to image
# what is params ?? cmp_data
#converts cmp files to image and take transpose of img array
def convert_to_images(params):
    context_width = 17
    n_frames, param_dim = params.shape
    border = context_width // 2

    params_with_borders = np.zeros(
        (2 * border + n_frames, param_dim), dtype=np.float32)
    params_with_borders[border:-border, :] = params
    params_with_borders[:border, :] = params[0, :]  # broadcast
    params_with_borders[-border:, :] = params[-1, :]  # broadcast

    params_with_borders_transposed = np.transpose(params_with_borders)

    params_as_images = np.zeros(
        (n_frames, param_dim, context_width, 1), dtype=np.float32)

    for i in range(n_frames):
        params_as_images[i, :, :,
                         0] = params_with_borders_transposed[:, i:i + context_width]

    return params_as_images
