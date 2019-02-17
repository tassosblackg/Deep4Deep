# script for reading each .cmp file from directories
# and then create images
import os
import glob as g
import numpy as np
import matplotlib.pyplot as plt

# very IMPORTANT to define the path ,dir_n
# check your project directories

path = "../../protocol_V2"               # check and define it properly
dir_n="/home/tassos/Desktop/DATA_ASR" # check and define where is your DATA dir

# read if .wav file is genuine or spoof --create label
# as filename put the full path of train_info file in protocolv2 dir
def read_label(filename):
    full_path = os.path.join(path, filename)
    class_type = []  # save labels
    with open(full_path, 'r') as f:
        # read line by line
        for line in f:
            l_words = line.split(" ")  # seperate words by space
            # files_n.append(l_words[0])
            if(l_words[1] == "genuine"):
                class_type.append(1)
            elif (l_words[1] == "spoof"):
                class_type.append(0)
            else:
                pass
            # print(class_type)
            # print("l="+str(len(class_type)))
    return class_type


# read  all files' name from directory and create an array of them
def read_cmp_dir(folder_name,n_files):
    files = g.glob(dir_n+"/"+folder_name + "/*.cmp")
    total_files = len(files)
    cmp_list = []
    if os.path.exists("read_status.txt"):
        f = open("read_status.txt","r")
        left_overs = int(f.readline())
        f.close()
    else:
        left_overs = total_files    # first read attempt


    start_i = total_files-left_overs # starting point of loop
    if (left_overs<n_files) :
        end_i = start_i + left_overs
        left_overs = 0
    else:
        end_i = start_i+n_files     # ending point
        left_overs=left_overs-n_files

    while (start_i<end_i):
        cmp_list.append(read_cmp_file(files[start_i]))
        start_i+=1
    if (left_overs != 0):
        # write left_overs in file for next session
        f = open("read_status.txt","w")
        f.write(str(left_overs))
        f.close()
    else:
        os.remove("read_status.txt")
        
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
    # print(cmp_data.shape)
    return (cmp_data)

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
    params_with_borders[:border, :] = params[0, :]    # broadcast
    params_with_borders[-border:, :] = params[-1, :]  # broadcast

    params_with_borders_transposed = np.transpose(params_with_borders)

    params_as_images = np.zeros(
        (n_frames, param_dim, context_width, 1), dtype=np.float32)

    for i in range(n_frames):
        params_as_images[i, :, :,
                         0] = params_with_borders_transposed[:, i:i + context_width]

    return params_as_images




# read all input image files of a directory
# and labels from each files
# args:
# @dir_name : directory name where .cmp files are saved
# @info_fl : text file where info about train,eval,dev sets are saved
def read_stage1(dir_name, info_fl,n_files):
    cmp_l = read_cmp_dir(dir_name,n_files)  # read .cmp files from dir
    # print("\ncmp_nl data have been read...\n")
    cl_types = read_label(info_fl)          # read label from info file
    # print("\ncl_types have been read...\n")
    data_l = []
    types=[]
    total_nframes=0;
    print("enter loop 1..\n")

    # for each cmp file
    for _ in range(cmp_l.__len__()):
        # cmp_data = read_cmp_file(cmp_nl[i])      # read that file
        cmp2img = convert_to_images(cmp_l.pop(0))  # convert this file to image --returns a np array
        nframes=cmp2img.shape[0] #size of np array
        #print(cmp2img.shape)                   # (num of images,heigt,width,1)
        types.extend([cl_types.pop(0)]*nframes) # instead to keep one label per cmp, keep for each frame of it
        total_nframes+=nframes                  # all images
        data_l.append(cmp2img)                  # keep all imgs -- a list with numpy array
    print("end of loop1..\n")
    # print(types[0])
    # print(len(types))
    del(cl_types)
    del(cmp_l)
    return data_l,types,total_nframes

# create a final numpy array of data and labels
def read_stage2(data,types,total_nframes):
    dim=data[0].shape[1]
    width=data[0].shape[2]

    # create all_params np array
    all_imgs=np.zeros(shape=(total_nframes,dim,width,1),dtype=np.float32) # initialize np array
    all_labels=np.zeros(shape=(total_nframes, ),dtype=np.int32)           # repeat labels type for each frame
    indx=0
    # iterate through list objects(numpy elements)
    for l in range(data.__len__()):
        cframes=data[0].shape[0]
        all_imgs[indx:indx+cframes,:]=data.pop(0)
        all_labels[indx:indx+cframes]=types.pop(0)
        indx=indx+cframes
    # print(all_imgs.shape)
    # print(all_labels.shape)
    # free space
    del(data)
    del(types)

    # all_data=np.concatenate((all_imgs,all_labels),axis=0)
    print("Data have been read...!\n")
    # save to file --run only once
    # np.save("Xdata",all_imgs)
    # np.save("Ydata",all_labels)
    return all_imgs,all_labels,total_nframes
    # return 10,100

# Read totally-- all files and labels
def read_Data(dir_name, info_fl,n_files):
    data,types,tframes=read_stage1(dir_name,info_fl,n_files)
    return(read_stage2(data,types,tframes))
    # return 10,100


# ------------------------------------------------------------------------------
