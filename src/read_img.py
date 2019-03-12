# script for reading each .cmp file from directories
# and then create images
import os
import glob as g
import numpy as np
import matplotlib.pyplot as plt

# very IMPORTANT to define the path ,dir_n
# check your project directories

path = "../protocol_V2"               # check and define it properly
dir_n="/home/tassos/Desktop/DATA_ASR" # check and define where is your DATA dir

# read if .wav file is genuine or spoof --create label
# as filename put the full path of train_info file in protocolv2 dir
# returns all encoded labels of the train data
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

    return class_type

# create one hot encoding labels
def one_hot_encode(labels):
    n_labels = len(labels) # how many labels per input data
    n_unique_labels = len(np.unique(labels)) # how many classes -- aka different labels
    print(n_labels,n_labels)
    ohe = np.zeros((n_labels,n_unique_labels))
    ohe[np.arange(n_labels),labels] = 1
    return ohe

# read  n_files .cmp files from a directory and create an array of them
# plus from a list of labels return a subset according the n_files
def read_cmp_dir(folder_name,class_types,n_files):
    files = g.glob(dir_n+"/"+folder_name + "/*.cmp")
    total_files = len(files)
    cmp_list = []
    labels_list = []
    read_status_file= folder_name+'_status.txt' # status file's name
    if os.path.exists(read_status_file):
        f = open(read_status_file,"r")
        left_overs = int(f.readline())
        f.close()
    else:
        left_overs = total_files    # first read attempt, no files have been read


    start_i = total_files-left_overs # starting point of loop
    if (left_overs<n_files) :
        end_i = start_i + left_overs
        left_overs = 0
    else:                           #
        end_i = start_i+n_files     # ending point
        left_overs=left_overs-n_files

    while (start_i<end_i): # read n append a subset of input data [start_i,end_i]
        cmp_list.append(read_cmp_file(files[start_i]))
        labels_list.append(class_types[start_i])
        start_i+=1
    if (left_overs != 0):
        # write left_overs in file for next session
        f = open(read_status_file,"w")
        f.write(str(left_overs))
        f.close()
    else:
        os.remove(read_status_file)

    return cmp_list,labels_list


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

    params_with_borders = np.zeros((2 * border + n_frames, param_dim), dtype=np.float32)
    params_with_borders[border:-border, :] = params
    params_with_borders[:border, :] = params[0, :]    # broadcast
    params_with_borders[-border:, :] = params[-1, :]  # broadcast

    params_with_borders_transposed = np.transpose(params_with_borders)

    params_as_images = np.zeros((n_frames, param_dim, context_width, 1), dtype=np.float32)

    for i in range(n_frames):
        params_as_images[i, :, :, 0] = params_with_borders_transposed[:, i:i + context_width]

    return params_as_images




# read all input image files of a directory
# args:
# @dir_name : directory name where .cmp files are saved
# @class_types  : a list with all labels of data set
# @n_files      : number of total files to be read
def read_stage1(dir_name, class_types,n_files):
    cmp_l,class_types = read_cmp_dir(dir_name,class_types,n_files)  # read (#n_files) .cmp files from dir
    # print("\ncmp_nl data have been read...\n")
    # cl_types = read_label(info_fl)          # read #n_files labels from info file
    # print("\ncl_types have been read...\n")
    data_l = []
    types=[]
    total_nframes=0;
    print("enter loop 1..\n")
    print('Cmp_l = ',str(cmp_l.__len__())+'\n')
    # for each cmp file
    for _ in range(cmp_l.__len__()):
        # cmp_data = read_cmp_file(cmp_nl[i])       # read that file
        cmp2img = convert_to_images(cmp_l.pop(0))   # convert this file to image --returns a np array
        nframes=cmp2img.shape[0]                    # size of np array
        #print(cmp2img.shape)                       # (num of images,heigt,width,1)
        types.extend([class_types.pop(0)]*nframes)  # instead to keep one label per cmp, keep for each frame of it
        total_nframes+=nframes                      # all images
        data_l.append(cmp2img)                      # keep all imgs -- a list with numpy array
    print("end of loop1..\n")
    # print(types[0],types[200])
    # print(len(types))
    del(class_types)
    del(cmp_l)
    return data_l,types,total_nframes

# create a final numpy array of data and labels
def read_stage2(data,types,total_nframes):
    dim=data[0].shape[1]    # 64
    width=data[0].shape[2]  # 17
    print("dim= ",str(dim)+"\n")
    print('width= ',str(width)+'\n')
    print('l= ',str(data[0].shape[0])+'\n')
    # create all_params np array
    all_imgs=np.zeros(shape=(total_nframes,dim,width,1),dtype=np.float32) # initialize np array -- all frames of a .cmp file
    all_labels=np.zeros(shape=(total_nframes, ),dtype=np.int32)           # each .cmp file many frames repeat label for number of frames
    indx=0
    # iterate through list objects(numpy elements)
    for l in range(data.__len__()):
        cframes=data[0].shape[0]
        print('cframes= ',str(cframes)+'\n')
        all_imgs[indx:indx+cframes,:]=data.pop(0)
        # print('data_pop= ',all_imgs[indx:indx+cframes,:])
        # print('\n')
        all_labels[indx:indx+cframes]=types.pop(0)
        # print('data_l= ',all_labels[indx:indx+cframes])
        # print('\n')
        indx=indx+cframes

    # free space
    del(data)
    del(types)
    # one-hot encode the labels
    print("all_l= ",all_labels)
    print("")
    print('L=',str(len(all_labels)))

    # save to file --run only once
    # np.save("Xdata",all_imgs)
    # np.save("Ydata",all_labels)
    return all_imgs,all_labels,total_nframes
    # return 10,100

# Read totally-- all files and labels
# @dir_name : directory name where .cmp files are saved
# @info_fl  : text file where info about train,eval,dev sets are saved
# @n_files  : how many files to be read
def read_Data(dir_name, info_file,n_files):
    class_types= read_label(info_file) # read all encoded-labels from a file
    ohe_l = one_hot_encode(class_types)
    # print("ohe_l= ",ohe_l)
    data,types,tframes=read_stage1(dir_name,ohe_l,n_files)
    return(read_stage2(data,types,tframes))
    # return 10,100


# ------------------------------------------------------------------------------
