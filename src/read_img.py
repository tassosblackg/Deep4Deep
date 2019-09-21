#----------------------------------------------------------------------------------------------------------------
#                                READ DATA & LABELS METHODs
#_________________________________________________________________________________________________________________

# script for reading each .cmp file from directories
# and then create images
import os
import glob as g
import numpy as np
import matplotlib.pyplot as plt

# very IMPORTANT to define the path ,dir_n
# check your project directories

path = "/home/tassos/Desktop/SpeechR2/Deep4Deep/protocol_V2"               # check and define it properly
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

# read a subset of labels
def read_subset_labels(file_name,n_files):
    labels_list = read_label(file_name) # file name of where labels are saved
    total_files = len(labels_list)
    subset_labels = []
    status_file_name = os.path.splitext(file_name)[0] + '_labels'
    start_i,end_i = read_status_upd(status_file_name,n_files,total_files)
    while (start_i<end_i): # read n append a subset of input data [start_i,end_i]
        subset_labels.append(labels_list[start_i]) # take one record
        start_i += 1

    del(labels_list)    # free up space

    return subset_labels

# create one hot encoding labels
def one_hot_encode(labels):
    n_labels = len(labels) # how many labels per input data
    # n_unique_labels = len(np.unique(labels)) # how many classes -- aka different labels
    n_unique_labels = 2
    print(n_labels,n_unique_labels)
    ohe = np.zeros((n_labels,n_unique_labels))
    ohe[np.arange(n_labels),labels] = 1
    return ohe

# keeps information about files left to be read
# helping for loop steps
# @filename : name of file depends
# @n_files : number of files to be read [==step of loop]
# @total_files : how many files per dataset
def read_status_upd(file_name,n_files,total_files):
    read_status_file= file_name+'_status.txt' # status file's name
    if os.path.exists(read_status_file):
        f = open(read_status_file,"r")
        left_overs = int(f.readline())
        # print(left_overs)
        f.close()
    else:
        left_overs = total_files    # first read attempt, no files have been read

    # calculate start - end index to read
    start_i = total_files-left_overs # starting point of loop
    if (left_overs<n_files) :
        end_i = start_i + left_overs
        left_overs -= n_files
    else:                           #
        end_i = start_i+n_files     # ending point
        left_overs=left_overs-n_files
    # save inforamtion of leftovers
    if (left_overs >= 0):
        # write left_overs in file for next session
        f = open(read_status_file,"w")
        f.write(str(left_overs))
        f.close()
    else:
        print('You read it.. " '+read_status_file+' " nothing left..! [ '+str(left_overs)+' ]\n')
        os.remove(read_status_file)

    return start_i,end_i

# read  n_files .cmp files from a directory and create an array of them
# plus from a list of labels return a subset according the n_files
def read_cmp_dir(folder_name,n_files):
    files = g.glob(dir_n+"/"+folder_name + "/*.cmp")    # read all files from a folder
    total_files = len(files)
    cmp_list = []
    start_i,end_i = read_status_upd(folder_name,n_files,total_files) # set start and end poind of reading
    print("start-end=\n")
    print(start_i,end_i)
    while (start_i<end_i): # read n append a subset of input data [start_i,end_i]
        cmp_list.append(read_cmp_file(files[start_i]))
        start_i += 1

    return cmp_list


# read a .cmp file and converted to numpy array (transposed)
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
    cmp_l = read_cmp_dir(dir_name,n_files)  # read (#n_files) .cmp files from dir
    # class_types to be defined ???
    data_l = []
    labels_per_frames = []
    total_nframes = 0
    print("enter loop 1..\n")
    print('Cmp_l = ',str(cmp_l.__len__())+'\n')
    # for each cmp file
    for i in range(len(cmp_l)):
        # cmp_data = read_cmp_file(cmp_nl[i])       # read that file
        cmp2img = convert_to_images(cmp_l.pop(0))   # convert this file to image --returns a np array
        nframes=cmp2img.shape[0]                    # size of np array
        #print(cmp2img.shape)                       # (num of images,heigt,width,1)
        labels_per_frames.extend([class_types[i]]*nframes)  # instead to keep one label per cmp, keep for each frame of it
        total_nframes+=nframes                      # all images
        data_l.append(cmp2img)                      # keep all imgs -- a list with numpy array
    print("end of loop1..\n")
    # print(types[0],types[200])
    # print(len(types))
    del(class_types)
    del(cmp_l)

    return data_l,labels_per_frames,total_nframes

# create a final numpy array of data and labels
# @data :  is a list with input data
# @labels : is a list with numpy array one-hot encoded labels []
# @total_nframes : is the total number of frames accross all read .cmp files together
def read_stage2(data,labels,total_nframes):
    Y = np.asarray(labels,dtype=np.int32)
    all_X=np.concatenate(data,axis=0)

    return all_X,Y,(len(Y))


# Read totally-- all files and labels
# @dir_name : directory name where .cmp files are saved
# @info_fl  : text file where info about train,eval,dev sets are saved
# @n_files  : how many files to be read
def read_Data(dir_name, info_file,n_files):
    class_types= read_subset_labels(info_file,n_files)              # read encoded-labels from a file 1st step
    ohe_l = one_hot_encode(class_types)                             # one -hot encode labels -2d step
    data,labels_l,tframes = read_stage1(dir_name,ohe_l,n_files)       # keep a subset of data in memory-3nd step
    #print(labels_l)
    np_data,np_labels,total_frames = read_stage2(data,labels_l,tframes)
    print("SHAPE= ")
    print(np_data.shape)
    # print('\ntransposed shape= ')
    # print((np_data.T).shape)
    print('\nlabels_shape= ')
    print(np_labels.shape)
    # reshaped_data = np_data.T
    in_nodes_dim = np_data.shape[1]*np_data.shape[2]*np_data.shape[3] # 64x17x1
    reshaped_data = np_data.reshape(tframes,in_nodes_dim) # reshape data to ndarray shape=(n_total_frames,in_nodes_dim)
    print('\nreshaped= ')
    print(reshaped_data.shape)

    return(reshaped_data,np_labels,total_frames)


# first attempt for read prediction data
def read_pred_data(dir_name,info_file):
    image,label,nframes = read_Data(dir_name,info_file,1) # read only one file
    droped_frames_image = image[:64,:]
    droped_labels = label[:64,:]
    return(droped_frames_image,droped_labels,64)

# ------------------------------------------------------------------------------
