function [y, extra, invhess] = mlpevfwd(net, x, t, x_test, invhess)
%MLPEVFWD Forward propagation with evidence for MLP

[y, z, a] = mlpfwd(net, x_test);
if nargin == 4
  [extra, invhess] = fevbayes(net, y, a, x, t, x_test);
else
  [extra, invhess] = fevbayes(net, y, a, x, t, x_test, invhess);
end
