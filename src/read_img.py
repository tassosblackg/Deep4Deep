# script for reading each .cmp file from directories
# and then create images
import os
import glob as g
import numpy as np
import matplotlib.pyplot as plt

#very IMPORTANT to define the right params_with_borders
#check your project directories

path = "../../protocol_V2"  # check and define it properly
dir_n="/home/tassos/Desktop/ASV/DATA" #check and define where is your DATA dir
# read if .wav file is genuine or spoof --create label
# as filename put the full path of train_info file in protocolv2 dir
def read_label(filename):
    full_path = os.path.join(path, filename)
    # files_n = [] #save .wav file's name
    class_type = []  # save labels
    with open(full_path, 'r') as f:
        # read line by line
        for line in f:
            l_words = line.split(" ")  # seperate words by space
            # files_n.append(l_words[0])
            if(l_words[1] == "genuine"):
                class_type.append('1')
            elif (l_words[1] == "spoof"):
                class_type.append('0')
            else:
                pass

    return class_type


# read  all files' name from directory and create an array of them
def read_cmp_dir(folder_name):
    files = g.glob(dir_n+"/"+folder_name + "/*.cmp")
    #print(files)
    cmp_list = []
    for i in range(files.__len__()):
        cmp_list.append(read_cmp_file(files[i]))
    #print(cmp_list.__len__())
    return cmp_list


# read .cmp a file and converted to numpy array (transposed)
def read_cmp_file(cmp_filename):
    nfilt = 64
    with open(cmp_filename, 'rb') as fid:
        cmp_data1 = np.fromfile(fid, dtype=np.float32, count=-1)

    cmp_data = cmp_data1.reshape((-1, nfilt))  # where nfilt = 64
    # @ check if read is done right
    # fbank = 30  # put a number between 0 and 63
    # plt.plot(cmp_data[:, fbank])
    # plt.show()
    return (cmp_data)


# read all input image files of a directory
# and labels from each files
# args:
# @dir_name : directory name where .cmp files are saved
# @info_fl : text file where info about train,eval,dev sets are saved
def read_Data(dir_name, info_fl):
    cmp_nl = read_cmp_dir(dir_name)  # read .cmp files from dir
    print("\ncmp_nl data have been read...\n")
    cl_types = read_label(info_fl)  # read label from info file
    print("\ncl_types have been read...\n")
    data_l = []
    types=[]
    total_nframes=0;
    dim=64
    print("enter loop 1..\n")
    # for each cmp file
    for i in range(cmp_nl.__len__()):
        # cmp_data = read_cmp_file(cmp_nl[i])  # read that file
        cmp2img = convert_to_images(cmp_data)  # convert this file to image --returns a np array
        nframes=cmp2img.shape[0] #size of np array
        types.append([cl_types[i]*nframes]) #instead to keep one label per cmp, keep for each frame of it
        total_nframes+=nframes #all images
        data_l.append(cmp2img) #keep all imgs -- a list with numpy array
    print("end of loop1..\n")
    #create all_params np array
    all_imgs=np.zeros(shape=(total_nframes,dim),dtype=np.float32) #initialize np array
    all_labels=np.zeros(shape=(total_nframes,1),dtype=np.int32) #repeat labels type for each frame
    indx=0
    print("in loop2..\n")
    #iterate through list objects(numpy elements)
    for l in range(data_l.__len__()):
        cframes=data_l[l].shape[0]
        all_imgs[indx:indx+nframes,:]=data_l[l]
        all_labels[indx:indx+nframes,:]=types[l]
        indx=indx+cframes
    print("end loop2..\n")
    #concatenate imgs with labels
    all_data=np.concatenate((all_imgs,all_labels),axis=0)
    print("Data have been read...!\n")


    return all_data,total_nframes

# ------------------------------------------------------------------------------
# convert .cmp file to image
# what is params ?? cmp_data
# converts cmp files to image and take transpose of img array


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
