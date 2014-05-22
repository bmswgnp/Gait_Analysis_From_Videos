function [y, extra, invhess] = glmevfwd(net, x, t, x_test, invhess)
%GLMEVFWD Forward propagation with evidence for GLM

[y, a] = glmfwd(net, x_test);
if nargin == 4
  [extra, invhess] = fevbayes(net, y, a, x, t, x_test);
else
  [extra, invhess] = fevbayes(net, y, a, x, t, x_test, invhess);
end
