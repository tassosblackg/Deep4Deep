#test reading .cmp files
import imp
imp.load_source("read_img","/home/tassos/Desktop/Deep4Deep/src/read_img.py")
import numpy as np
import matplotlib.pyplot as plt
import read_img as rim

rim.read_cmp_file("testRead/T_1000001.cmp")
