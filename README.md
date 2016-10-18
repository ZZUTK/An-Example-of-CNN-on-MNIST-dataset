# An Example of CNN on MNIST dataset

Detailes of the CNN strucdure in the demo, as well as the mathematical derivation of 
backpropagation can be found in ["Derivation of Backpropagation in Convolutional Neural Network (CNN)"](), which is specifically written for this demo.  

The implementation of CNN uses the trimmed version of DeepLearnToolbox by [R. B. Palm](https://github.com/rasmusbergpalm/DeepLearnToolbox). 

## Pre-requisite
* Matlab

## CNN structure in the demo
<img src="https://github.com/ZZUTK/An-Example-of-CNN-on-MNIST-dataset-/blob/master/figs/CNN.png">

## Run the demo

```
>> demo_CNN_MNIST
```

Note that only 1 epoch will be performs. If you want to run more epochs, please modify the variable `num_epochs ` in the file [`demo_CNN_MNIST.m`](https://github.com/ZZUTK/An-Example-of-CNN-on-MNIST-dataset-/blob/master/demo_CNN_MNIST.m) (line 62).

## Results
Run the demo for 200 epochs, the classification accuracy will be over 1%. The class-wise classification accuracy is shown as follow.
<img src="https://github.com/ZZUTK/An-Example-of-CNN-on-MNIST-dataset-/blob/master/figs/class-wise_accuracy.jpg", width="400", hspace="200">

### Training and testing accuracy
<img src="https://github.com/ZZUTK/An-Example-of-CNN-on-MNIST-dataset-/blob/master/figs/train_accu.png", width="400">
<img src="https://github.com/ZZUTK/An-Example-of-CNN-on-MNIST-dataset-/blob/master/figs/test_accu.png", width="400">

### Tracing error in Mean Square Error (MSE) 
<img src="https://github.com/ZZUTK/An-Example-of-CNN-on-MNIST-dataset-/blob/master/figs/train_MSE.png", width="400">

### The learned kernels for the first and second convolutional layers
The first convolutional layer has 6 kernels, and the second has 6x12 kernels. All kernels are in the size of 5x5.
<img src="https://github.com/ZZUTK/An-Example-of-CNN-on-MNIST-dataset-/blob/master/figs/kernels_Conv1.jpg", width="400">
<img src="https://github.com/ZZUTK/An-Example-of-CNN-on-MNIST-dataset-/blob/master/figs/kernels_Conv2.jpg", width="400">

###  
