function net = mlpinit(net, prior)
%MLPINIT Initialise the weights in a 2-layer feedforward network.

if isstruct(prior)
  sig = 1./sqrt(prior.index*prior.alpha);
  w = sig'.*randn(1, net.nwts); 
elseif size(prior) == [1 1]
  w = randn(1, net.nwts).*sqrt(1/prior);
else
  error('prior must be a scalar or a structure');
end  

net = mlpunpak(net, w);

