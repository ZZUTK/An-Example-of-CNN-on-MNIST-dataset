%% Demo of training and testing CNN on the MNIST dataset
% Based on the DeepLearnToolbox by R. B. Palm
% Written by Zhifei Zhang, 8/19/2016
% University of Tennessee, Knoxville, TN
% Tested on Windows 8.1 Matlab R2015a

clc; clear; close all;

%% add path of DeepLearnToolbox
addpath(genpath('DeepLearnToolbox_trimmed'));

%% load MNIST dataset
% 60,000 training images in size of 28x28
% 10,000 testing images in size of 28x28
% 10 categories (0~9) with one-hot label
load mnist_uint8;

%% format the data
% images (*_x): [width, height, num_images]
% labels (*_y): [num_classes, num_images]
train_x = double(reshape(train_x',28,28,[]))/255;
train_x = permute(train_x, [2 1 3]);
train_y = double(train_y');

test_x = double(reshape(test_x',28,28,[]))/255;
test_x = permute(test_x, [2 1 3]);
test_y = double(test_y');

%% setup a convolutional neural network
% input (map size: 28x28)
%   --> convolution with 6 kernels of size 5x5 + sigmoid (24x24x6)
%   --> subsampling with 2x2 kernel (12x12x6)
%   --> convolution with 6x12 kernels of size 5x5 + sigmoid (8x8x12)
%   --> subsampling with 2x2 kernel (4x4x12) + vectorization (192x1)
%   --> fully connection + sigmoid => output (10x1)
cnn.layers = {
    %input layer
    struct('type', 'i') 
    
    %convolution layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) 
    
    %subsampling layer (average pooling)
    struct('type', 's', 'scale', 2) 
    
    %convolution layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) 
    
    %subsampling layer (average pooling)
    struct('type', 's', 'scale', 2) 
};
cnn = cnnsetup(cnn, train_x, train_y);

%% train the CNN
% learning rate: the rate of updating the kernels
learning_rate = 1; 
% batch size: the number of images inputed to the network for each update  
batch_size = 50; 
% number of epochs: iterations of training on the whole dataset
% 1 epoch gets around 11% error rate, and 100 epochs gets around 1.2%
% each epoch takes around 90s on Intel dual core i7 CPU @ 2.40GHz 
num_epochs = 200; 

opts.alpha = learning_rate;
opts.batchsize = batch_size;
opts.numepochs = num_epochs;

% use the classical gradient descent in backpropagation
cnn = cnntrain(cnn, train_x, train_y, opts);

%% test the CNN
[error_rate, cnn, bad] = cnntest(cnn, test_x, test_y);
fprintf('Error rate = %.2f%%\n', error_rate*100);

%% training and testing is done!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% later code is for plotting %%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% plot batch-wise mean squared error during training 
figure('numberTitle', 'off', 'name', ...
    'Batch-wise mean squared error during training'); 
plot(cnn.rL, 'linewidth', 2);
xlabel('Batches')
ylabel('Mean squared error')
set(gca, 'fontsize', 16, ...
    'xlim', [0 length(cnn.rL)], ...
    'ylim', [0 ceil(max(cnn.rL)*10)*.1]);
grid on

%% the class-wise classification accuracy
[~, labels] = max(test_y);
h1 = hist(labels, size(test_y,1));
labels(bad) = [];
h2 = hist(labels, size(test_y,1));
figure('numberTitle', 'off', 'name', ...
    'Class-wise classification accuracy'); hold on
colors = get(gca, 'colororder');
bar(0:9, h1, .85, 'facecolor', colors(1,:))
bar(0:9, h2, .80, 'facecolor', colors(2,:))
legend('Number of testing samples', ...
    'Number of right classified samples', 'location', 'southeast')
set(gca, ...
    'ygrid', 'on', ...
    'fontsize', 14, ...
    'xlim', [-.5 9.5], ...
    'xtick', 0:9, ...
    'ylim', [0 1200] ...
    )
accuracy = h2 ./ h1;
for i = 1:length(accuracy)
    str = sprintf('%.2f', accuracy(i));
    text(i-1.4, h1(i)+30, str, 'fontsize', 14);
end

%% visualize the convolution kernels
for i = 1:length(cnn.layers)
    if cnn.layers{i}.type ~= 'c'
        continue
    end
    kernels = cnn.layers{i}.k;
    num_input_maps = length(kernels);
    num_output_maps = length(kernels{1});
    figure('numberTitle', 'off', 'name', ...
        sprintf('Kernels of the Conv. layer: %d input -> %d output', ...
        num_input_maps, num_output_maps));
    for j = 1:num_input_maps
        for k = 1:num_output_maps
            subplot(num_input_maps, num_output_maps, ...
                (j-1)*num_output_maps+k)
            imagesc(kernels{j}{k})
            colormap gray
            axis image
            set(gca, ...
                'xticklabel', [], ...
                'yticklabel', [], ...
                'ticklength', [0 0])
        end
    end
end

%% an example of feedforward using the trained network
figure('numberTitle', 'off', 'name', ...
    'An example of feed forward');
num_layers = length(cnn.layers);
n_rows = length(cnn.layers{end}.b);
n_cols = num_layers + 2;
sample_ind = 1; % 1~batch_size
value_range = [0 1];
for i = 1:num_layers
    switch cnn.layers{i}.type
        % plot imput image
        case 'i'
            subplot(n_rows, n_cols, 1:n_cols:(n_rows*n_cols))
            img = cnn.layers{i}.a{1}(:,:,sample_ind);
            title_str = 'Input';
            imagesc(img, value_range);
            colormap gray
            axis image
            set(gca, ...
                'xticklabel', [], ...
                'yticklabel', [], ...
                'ticklength', [0 0])
            title(title_str, 'fontsize', 14)
            
        % plot maps after convolution
        case 'c'
            img_size = size(cnn.layers{i}.a{1});
            title_str = 'Conv';
            span = n_rows / length(cnn.layers{i}.a);
            for j = 1:length(cnn.layers{i}.a)
                locs = [];
                for k = 1:span 
                    locs = [locs (j-1)*span*n_cols+(k-1)*n_cols+i];
                end
                subplot(n_rows, n_cols, locs)
                img = cnn.layers{i}.a{j}(:,:,sample_ind);
                imagesc(img, value_range);
                colormap gray
                axis image
                set(gca, ...
                    'xticklabel', [], ...
                    'yticklabel', [], ...
                    'ticklength', [0 0])
                if j == 1
                    title(title_str, 'fontsize', 14)
                end
            end
                
        % plot maps after pooling        
        case 's'
            img_size = size(cnn.layers{i}.a{1});
            title_str = 'Pool';
            span = n_rows / length(cnn.layers{i}.a);
            for j = 1:length(cnn.layers{i}.a)
                locs = [];
                for k = 1:span 
                    locs = [locs (j-1)*span*n_cols+(k-1)*n_cols+i];
                end
                subplot(n_rows, n_cols, locs)
                img = cnn.layers{i}.a{j}(:,:,sample_ind);
                imagesc(img, value_range);
                colormap gray
                axis image
                set(gca, ...
                    'xticklabel', [], ...
                    'yticklabel', [], ...
                    'ticklength', [0 0])
                if j == 1
                    title(title_str, 'fontsize', 14)
                end
            end
    end
end

% plot fully-connected layer
subplot(n_rows, n_cols, (n_cols-1):n_cols:(n_rows*n_cols))        
img = cnn.fv(:,sample_ind);
imagesc(img, value_range);
colormap gray
axis image
set(gca, ...
    'xticklabel', [], ...
    'yticklabel', [], ...
    'ticklength', [0 0])
title('FC', 'fontsize', 14)

% plot output layer
subplot(n_rows, n_cols, n_cols:n_cols:(n_rows*n_cols))        
img = cnn.o(:,sample_ind);
imagesc(img, value_range);
colormap gray
axis image
set(gca, ...
    'xticklabel', [], ...
    'yticklabel', 0:9, ...
    'ticklength', [0 0], ...
    'fontsize', 14, ...
    'ytick', 1:10 ...
    )
title('Output', 'fontsize', 14)
