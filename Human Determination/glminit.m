function net = glminit(net, prior)
%GLMINIT Initialise the weights in a generalized linear model.


if ~strcmp(net.type, 'glm')
  error('Model type should be ''glm'');
end
if isstruct(prior)
  sig = 1./sqrt(prior.index*prior.alpha);
  w = sig'.*randn(1, net.nwts); 
elseif size(prior) == [1 1]
  w = randn(1, net.nwts).*sqrt(1/prior);
else
  error('prior must be a scalar or a structure');
end  

net = glmunpak(net, w);

