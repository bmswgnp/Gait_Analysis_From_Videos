function net = netinit(net, prior)
%NETINIT Initialise the weights in a network.

if isstruct(prior)
    if (isfield(net, 'mask'))
	if find(sum(prior.index, 2)) ~= find(net.mask)
	    error('Index does not match mask');
	end
	sig = sqrt(prior.index*prior.alpha);
	% Weights corresponding to zeros in mask will not be used anyway
	% Set their priors to one to avoid division by zero
	sig = sig + (sig == 0);  
	sig = 1./sqrt(sig);
    else
	sig = 1./sqrt(prior.index*prior.alpha);
    end
    w = sig'.*randn(1, net.nwts); 
elseif size(prior) == [1 1]
  w = randn(1, net.nwts).*sqrt(1/prior);
else
  error('prior must be a scalar or a structure');
end  

if (isfield(net, 'mask'))
    w = w(logical(net.mask));
end
net = netunpak(net, w);

