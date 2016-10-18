%% Demo of training and testing CNN on simple synthatic images
% Based on the DeepLearnToolbox by R. B. Palm
% Written by Zhifei Zhang, 10/4/2016
% University of Tennessee, Knoxville, TN
% Tested on Windows 8.1 Matlab R2015a

clc; clear; close all;

%% add path of DeepLearnToolbox
addpath(genpath('DeepLearnToolbox_trimmed'));

%% generate traing and testing images
num_classes = 3;  
num_images_per_class = 3; 
% simple images reshaped into columns 
% image size (3x3) reshaped to (9x1)
images = [
    % class 1
    255     0   255   255     0   255   255     0   255
    198     5   213   244    14   245   241     8   231    
    246     8   222   225    40   237   228     5   235
    
    % class 2
    255   255   255   255     0   255   255   255   255
    234   255   205   251     0   251   238   253   240    
    232   255   231   247    38   246   190   236   250
    
    % class 3
    255    255   255    0   0   0   255    255   255
    245    225   205    1   0   5   238    253   240    
    225    235   231    7   8   4   190    236   250
    ]';

% labels
labels = [  
    1  1   1   0   0   0   0   0   0
    0  0   0   1   1   1   0   0   0
    0  0   0   0   0   0   1   1   1 
    ];

%% format the data
train_x = reshape(images, 3, 3, []) / 255.0;
train_x = imresize(train_x, [6 6]);
train_y = labels;

test_x = reshape(images, 3, 3, []) / 255.0;
test_x = imresize(test_x, [6 6]);
test_x = imnoise(test_x, 'gaussian'); % add gaussian noise
test_y = labels;

% display the training images 
figure('numberTitle', 'off', 'name', ...
    'Training images');
for i=1:num_images_per_class*num_classes
    subplot(num_classes,num_images_per_class,i), ...
        imshow(train_x(:,:,i))
    axis on
    set(gca, ...
        'xticklabel', [], ...
        'yticklabel', [], ...
        'ticklength', [0 0])
    title(strcat('Train image / Class ', int2str(ceil(i/num_images_per_class))))
end

% display the testing images 
figure('numberTitle', 'off', 'name', ...
    'Testing images');
for i=1:num_images_per_class*num_classes
    subplot(num_classes,num_images_per_class,i), ...
        imshow(test_x(:,:,i))
    axis on
    set(gca, ...
        'xticklabel', [], ...
        'yticklabel', [], ...
        'ticklength', [0 0])
    title(strcat('Test image ', int2str(i)))
end

%% setup a convolutional neural network
% input (map size: 6x6)
%   --> convolution with 2 kernels of size 3x3 + sigmoid (4x4x2)
%   --> subsampling with 2x2 kernel (2x2x2)
%   --> vectorization (8x1)
%   --> fully connection + sigmoid => output (3x1)
rng(0,'v5uniform') % reset the rand generator
cnn.layers = {
    %input layer
    struct('type', 'i') 
    
    %convolution layer
    struct('type', 'c', 'outputmaps', 2, 'kernelsize', 3) 
    
    %subsampling layer (average pooling)
    struct('type', 's', 'scale', 2) 
};
cnn = cnnsetup(cnn, train_x, train_y);

%% train the CNN
% learning rate: the rate of updating the kernels
learning_rate = 1; 
% batch size: the number of images inputed to the network for each update  
batch_size = 9; 
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
bar(1:3, h1, .8, 'facecolor', colors(1,:))
bar(1:3, h2, .7, 'facecolor', colors(2,:))
legend('Number of testing samples', ...
    'Number of right classified samples', 'location', 'southeast')
set(gca, ...
    'ygrid', 'on', ...
    'fontsize', 14, ...
    'xlim', [.5 3.5], ...
    'xtick', 1:3, ...
    'ylim', [0 3.5] ...
    )
accuracy = h2 ./ h1;
for i = 1:length(accuracy)
    str = sprintf('%.2f', accuracy(i));
    text(i-.065, h1(i)+.2, str, 'fontsize', 14);
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
        % plot input images
        case 'i'
            subplot(n_rows, n_cols, 1:n_cols:(n_rows*n_cols))
            img = cnn.layers{i}.a{1}(:,:,sample_ind);
            [m, n] = size(img);
            title_str = {'Input', sprintf('(%dx%d)', m, n)};
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
                    [m, n] = size(img);
                    title_str = {'Conv', sprintf('(%dx%d)', m, n)};
                    title(title_str, 'fontsize', 14)
                end
            end
                
        % plot maps after pooling    
        case 's'
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
                    [m, n] = size(img);
                    title_str = {'Pool', sprintf('(%dx%d)', m, n)};
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
title_str = ['FC ', sprintf('(%dx1)', length(img))];
title(title_str, 'fontsize', 14)

% plot output layer
subplot(n_rows, n_cols, (n_cols):n_cols:(n_rows*n_cols))        
img = cnn.o(:,sample_ind);
imagesc(img, value_range);
colormap gray
axis image
title_str = {'Output ', '(3x1)'};
title(title_str, 'fontsize', 14)
set(gca, 'xticklabel', [], ...
    'ytick', 1:3, ...
    'yticklabel', 1:3, ...
    'fontsize', 14, ...
    'ticklength', [0 0])
