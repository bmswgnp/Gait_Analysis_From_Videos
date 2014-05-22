function [h, hdata] = mlphess(net, x, t, hdata)
%MLPHESS Evaluate the Hessian matrix for a multi-layer perceptron network.

% Check arguments for consistency
errstring = consist(net, 'mlp', x, t);
if ~isempty(errstring);
  error(errstring);
end

if nargin == 3
  % Data term in Hessian needs to be computed
  hdata = datahess(net, x, t);
end

[h, hdata] = hbayes(net, hdata);

% Sub-function to compute data part of Hessian
function hdata = datahess(net, x, t)

hdata = zeros(net.nwts, net.nwts);

for v = eye(net.nwts);
  hdata(find(v),:) = mlphdotv(net, x, t, v);
end

return
