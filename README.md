# An Example of CNN on MNIST dataset

Detailes of the CNN strucdure in the demo, as well as the mathematical derivation of 
backpropagation can be found in ["Derivation of Backpropagation in Convolutional Neural Network (CNN)"](), which is specifically written for this demo.  

The implementation of CNN uses the trimmed version of DeepLearnToolbox by [R. B. Palm](https://github.com/rasmusbergpalm/DeepLearnToolbox). 

## Contents
* [Pre-requisite](#Requirements)
* [MNIST dataset](#MNIST dataset)
 * [Some samples of digit images](#samples)
* [CNN structure](#CNN structure)
* [Run the demo](#Run)
* [Results](#Results)

<a name="Requirements">
## Pre-requisite
* Matlab

<a name="MNIST dataset">
## MNIST dataset
In the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset, there are 50,000 digit images for training and 10,000 for testing. The image size is 28x28, and the digits are from 0 to 9 (10 categories). 

<a name="samples">
### Some samples of digit images
<img src="https://github.com/ZZUTK/An-Example-of-CNN-on-MNIST-dataset-/blob/master/figs/digits.png">

<a name="CNN structure">
## CNN structure in the demo
<img src="https://github.com/ZZUTK/An-Example-of-CNN-on-MNIST-dataset-/blob/master/figs/CNN.png">

<a name="Run">
## Run the demo

```
>> demo_CNN_MNIST
```

Note that only 1 epoch will be performs. If you want to run more epochs, please modify the variable `num_epochs ` in the file [`demo_CNN_MNIST.m`](https://github.com/ZZUTK/An-Example-of-CNN-on-MNIST-dataset-/blob/master/demo_CNN_MNIST.m) (line 62).

<a name="Results">
## Results
Running the demo for 200 epochs, the classification accuracy is shown as follow. Note that the results may be a little bit different for each running because of the random initialization of convolutional kernels.

| num_epochs | Training accuracy | Testing accuracy |
| :---: | :---: | :---: |
| 200 | 99.34% | 99.02% |

<a name="Class-wise">
### Class-wise training and testing accuracy
<img src="https://github.com/ZZUTK/An-Example-of-CNN-on-MNIST-dataset-/blob/master/figs/class-wise_accu_train.png", width="400">
<img src="https://github.com/ZZUTK/An-Example-of-CNN-on-MNIST-dataset-/blob/master/figs/class-wise_accu_test.png", width="400">

<a name="accuracy">
### Training and testing accuracy
<img src="https://github.com/ZZUTK/An-Example-of-CNN-on-MNIST-dataset-/blob/master/figs/train_accu.png", width="400">
<img src="https://github.com/ZZUTK/An-Example-of-CNN-on-MNIST-dataset-/blob/master/figs/test_accu.png", width="400">

<a name="Error">
### Training error in Mean Square Error (MSE) 
The loss function used in this demo is 

<img src="https://github.com/ZZUTK/An-Example-of-CNN-on-MNIST-dataset-/blob/master/figs/loss_func.png", width="200">

where y and y_hat denote the true label and prediction, respectively.

<img src="https://github.com/ZZUTK/An-Example-of-CNN-on-MNIST-dataset-/blob/master/figs/train_MSE.png", width="400">

<a name="kernels">
### The learned kernels of the first and second convolutional layers
The first convolutional layer has 6 kernels, and the second has 6x12 kernels. All kernels are in the size of 5x5.
<img src="https://github.com/ZZUTK/An-Example-of-CNN-on-MNIST-dataset-/blob/master/figs/kernels_Conv1.jpg", width="400">
<img src="https://github.com/ZZUTK/An-Example-of-CNN-on-MNIST-dataset-/blob/master/figs/kernels_Conv2.jpg", width="400">

<a name="example">
### An example of feedforward on the trained CNN
<img src="https://github.com/ZZUTK/An-Example-of-CNN-on-MNIST-dataset-/blob/master/figs/example_feedforward.jpg">
