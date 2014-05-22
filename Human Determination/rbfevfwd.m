function [y, extra, invhess] = rbfevfwd(net, x, t, x_test, invhess)
%RBFEVFWD Forward propagation with evidence for RBF

y = rbffwd(net, x_test);
% RBF outputs must be linear, so just pass them twice (second copy is 
% not used
if nargin == 4
  [extra, invhess] = fevbayes(net, y, y, x, t, x_test);
else
  [extra, invhess] = fevbayes(net, y, y, x, t, x_test, invhess);    
end
