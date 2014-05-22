function w = rbfpak(net)
%RBFPAK	Combines all the parameters in an RBF network into one weights vector.

errstring = consist(net, 'rbf');
if ~errstring
  error(errstring);
end

w = [net.c(:)', net.wi, net.w2(:)', net.b2];
