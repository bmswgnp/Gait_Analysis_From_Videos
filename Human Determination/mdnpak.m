function w = mdnpak(net)
%MDNPAK	Combines weights and biases into one weights vector.

errstring = consist(net, 'mdn');
if ~errstring
  error(errstring);
end
w = mlppak(net.mlp);
