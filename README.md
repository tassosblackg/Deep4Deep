# Deep4Deep
**CNN** implementation for ASV using tensorflow.

Anti-Spoofing speech recognition classification (based on ASV2017 contest).
__________________________________________________________________________________________________________________________
**_Requirements:_**
-----------------

Python 2.7 or higher

- **Tensorflow** Framework (build from source preffered)
- **SoundFile** 
  > pip install SoundFile
- **Librosa**
  > pip install librosa
- **matplotlib**
  > pip install matplotlib
- **Tkinter** 
  > sudo apt-get install pythonX-tk , *where X is the needed python version 2,3*

_________________________________________________________________________________________________________________________
**_Preprocessing:_**
--------------------
Create filter banks [*windows from sound files*] for each data set.

> run make_fbanks.py *settype*, *path*
  - settype arg: *'train', 'dev', 'eval'* for each dataset
  - path arg: path to where data set are

**_Filter Banks:_** are an alterantive way for sampling a signal and get usefull coefficients as features of the signal. 

With this technique we sample the signal and group past,current and future windows with a specific sampling-period for the parts of the signal 
and build frames.These frames will be used to feed them in our CNN model.

*Other feature extraction technique for sound signal is MFCC.*
 
**More infos:**

 > *Check projectDescription in Docs*
_________________________________________________________________________________________________________________________
**_Architecture:_**
-------------------
In this project we implement a *10-layer CNN* for classification of a speech file(.wav) into 2 classes (*Genuine or Spoof*)

- We have up to 5 blocks, each block has *2 convolution + 1 max_pooling layer*

 * Apply batch_normalization after each convolutional layer

- We use up to *64 outputchannels per layer*(instead of 256, that is proposed on vd10-fpad-tpad)..


- Last is Dense layers (flatten, ReLU,softmax) : 

**More Infos about model:**
  >  *see: VeryDeepCNN.pdf in docs/*
_____________________________________________________________________________________________________________________________

_DataSet:_
----------
All data are labeled with *spoof or genuine*. They also have taken with different methods and enviroments. 
- **Train Set:** 
- **Dev Set:**
- **Eval Set:**

> *see more : *
_____________________________________________________________________________________________________________________________
**_Results:_**
---------------
Still waiting for a decent GPU to train and test the model..

Because it's impossible to train and evaluate this model with a mid-range laptop's CPU

Soon It will be updated

We compare our results with those referred at

docs/ Kinnunen_2017_The_ASVspoof_2017_challenge_Assessing_the_limits_of_replay_spoofing_attack_detection.pdf

______________________________________________________________________________________________________
*Usefull Links:*
----------------
> *https://datashare.is.ed.ac.uk/handle/10283/3055*
