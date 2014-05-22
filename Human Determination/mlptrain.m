function [net, error] = mlptrain(net, x, t, its);
%MLPTRAIN Utility to train an MLP network for demtrain

options = zeros(1,18);
options(1) = -1;	% To prevent any messages at all
options(9) = 0;
options(14) = its;

[net, options] = netopt(net, options, x, t, 'scg');

error = options(8);

