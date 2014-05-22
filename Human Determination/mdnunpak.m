function net = mdnunpak(net, w)
%MDNUNPAK Separates weights vector into weight and bias matrices. 

errstring = consist(net, 'mdn');
if ~errstring
  error(errstring);
end
if net.nwts ~= length(w)
  error('Invalid weight vector length')
end

net.mlp = mlpunpak(net.mlp, w);
