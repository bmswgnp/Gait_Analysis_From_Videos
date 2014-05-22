function w = netpak(net)
%NETPAK	Combines weights and biases into one weights vector.

pakstr = [net.type, 'pak'];
w = feval(pakstr, net);
% Return masked subset of weights
if (isfield(net, 'mask'))
   w = w(logical(net.mask));
end
