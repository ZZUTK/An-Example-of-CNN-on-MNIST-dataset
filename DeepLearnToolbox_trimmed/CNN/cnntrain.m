function net = cnntrain(net, x, y, opts)
    m = size(x, 3);
    numbatches = m / opts.batchsize;
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end
    net.rL = [];
    h = waitbar(0, '', 'name', 'Training CNN ...');
    for i = 1 : opts.numepochs % for each epoch
        fprintf('Epoch\t%2d/%2d\n', i, opts.numepochs);
        start_epoch = tic;
        kk = randperm(m);
        for l = 1 : numbatches % for each batch
            waitbar(((i-1)*numbatches+l) / (numbatches*opts.numepochs), ...
                h, sprintf('epoch %d/%d, batch %d/%d', ...
                i, opts.numepochs, l, numbatches))
            
            batch_x = x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            batch_y = y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            % feed forward
            net = cnnff(net, batch_x);
            % back propagation (gradient descent)
            net = cnnbp(net, batch_y);
            % update parameters 
            net = cnnapplygrads(net, opts);
            if isempty(net.rL)
                net.rL(1) = net.L;
            end
            net.rL(end + 1) = .9 * net.rL(end) + .1 * net.L;
        end
        toc(start_epoch)
    end
    close(h)
end
