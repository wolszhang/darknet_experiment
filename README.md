# darknet_experiment
In this repo, the neural network framework we use are adopted from https://github.com/pjreddie/darknet/blob/master/cfg/yolov2.cfg

We conduct several experiment involving using the darknet architecture for transfer learning
measure some benchmark testing using darknet

## Introduction

You may think "darknet" as the hidden part of the internet, but it's actually also name of a open source neural network framework written in C and CUDA which is fast (because of the underlying C is optimized), easy to use, and support CPU and GPU computation. In this blog, I will introduce what darknet supports and test several benchmarks using darknet framework.

## Installation and Preparation:
First of all, you should follow the installation tutorial here[link](https://pjreddie.com/darknet/install/). For user with GPU and CUDA, you could enable GPU computation by changing the first line in the makefile to 
'GPU=1'

Darknet also support openCV for more image types by changing the second line in the makefile to 
'OPENCV=1'

You can check your installation by trying
'./darknet' 
in your current terminal, which should pop up a window like 
