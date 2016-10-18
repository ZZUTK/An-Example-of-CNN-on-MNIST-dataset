function [err_rate, net, bad, est] = cnntest(net, x, y)
    %  feedforward
    net = cnnff(net, x);
    [~, est] = max(net.o);
    [~, tru] = max(y);
    bad = find(est ~= tru);

    err_rate = numel(bad) / size(y, 2);
end
