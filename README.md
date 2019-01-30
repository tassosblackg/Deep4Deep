# Deep4Deep
CNN implementation for ASV using tensorflow

Anti-Spoofing speech recognition classification (based on ASV2017 contest)
------------------------------------------------------------------------------
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

-------------------------------------------------------------------------------
**_Architecture:_**
-------------------
In this project we implement a 10-layer CNN for classification of a speech file(.wav) into 2 classes (Genuine or Spoof)

- We have up to 5 blocks, each block has 2 convolution + 1 max_pooling layer 

 * Apply batch_normalization after each convolutional layer

- We use up to 64 outputchannels per layer(instead of 256, that is proposed on vd10-fpad-tpad)..

see: VeryDeepCNN.pdf in docs/

- Last is Dense layers (flatten, ReLU,softmax) : 

_DataSet:_
----------

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
