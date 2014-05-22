function [e, edata, eprior] = rbferr(net, x, t)
%RBFERR	Evaluate error function for RBF network.

% Check arguments for consistency
switch net.outfn
case 'linear'
   errstring = consist(net, 'rbf', x, t);
case 'neuroscale'
   errstring = consist(net, 'rbf', x);
otherwise
   error(['Unknown output function ', net.outfn]);
end
if ~isempty(errstring);
  error(errstring);
end

switch net.outfn
case 'linear'
   y = rbffwd(net, x);
   edata = 0.5*sum(sum((y - t).^2));
case 'neuroscale'
   y = rbffwd(net, x);
   y_dist = sqrt(dist2(y, y));
   % Take t as target distance matrix
   edata = 0.5.*(sum(sum((t-y_dist).^2)));
otherwise
   error(['Unknown output function ', net.outfn]);
end

% Compute Bayesian regularised error
[e, edata, eprior] = errbayes(net, edata);

