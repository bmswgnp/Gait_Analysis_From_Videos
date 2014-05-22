function [g, gdata, gprior] = mlpgrad(net, x, t)
%MLPGRAD Evaluate gradient of error function for 2-layer network.

% Check arguments for consistency
errstring = consist(net, 'mlp', x, t);
if ~isempty(errstring);
  error(errstring);
end
[y, z] = mlpfwd(net, x);
delout = y - t;

gdata = mlpbkp(net, x, z, delout);

[g, gdata, gprior] = gbayes(net, gdata);
