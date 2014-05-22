function net = gpinit(net, tr_in, tr_targets, prior)
%GPINIT	Initialise Gaussian Process model.

errstring = consist(net, 'gp', tr_in, tr_targets);
if ~isempty(errstring);
  error(errstring);
end

if nargin >= 4 
  % Initialise weights at random
  if size(prior.pr_mean) == [1 1]
    w = randn(1, net.nwts).*sqrt(prior.pr_var) + ...
       repmat(prior.pr_mean, 1, net.nwts);
  else
    sig = sqrt(prior.index*prior.pr_var);
    w = sig'.*randn(1, net.nwts) + (prior.index*prior.pr_mean)'; 
  end
  net = gpunpak(net, w);
end

net.tr_in = tr_in;
net.tr_targets = tr_targets;
