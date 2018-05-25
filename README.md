# Deep4Deep
CNN implementation for ASV using tensorflow

Anti-Spoofing speech recognition classification (based on ASV2017 contest)

-------------------------------------------------------------------------------
@Architecture:
--------------
In this project we implement a 10-layer CNN for classification of a speech file(.wav) into 2 classes (Genuine or Spoof)

-We have up to 5 blocks, each block has 2 convolution + 1 max_pooling layer 

*(->apply batch_normalization after each convolutional layer)

-We use up to 64 outputchannels per layer(instead of 256, that is proposed on vd10-fpad-tpad)..

see: VeryDeepCNN.pdf in docs/

-Last is Dense layers (flatten, ReLU,softmax) : 

@DataSet:
---------

@Results:
---------
We compare our results with those referred at

docs/ Kinnunen_2017_The_ASVspoof_2017_challenge_Assessing_the_limits_of_replay_spoofing_attack_detection.pdf

______________________________________________________________________________________________________
Usefull Links:
---------------
