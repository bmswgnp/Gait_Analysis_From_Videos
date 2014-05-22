function net = netunpak(net, w)
%NETUNPAK Separates weights vector into weight and bias matrices. 

unpakstr = [net.type, 'unpak'];

% Check if we are being passed a masked set of weights
if (isfield(net, 'mask'))
   if length(w) ~= size(find(net.mask), 1)
      error('Weight vector length does not match mask length')
   end
   % Do a full pack of all current network weights
   pakstr = [net.type, 'pak'];
   fullw = feval(pakstr, net);
   % Replace current weights with new ones
   fullw(logical(net.mask)) = w;
   w = fullw;
end

net = feval(unpakstr, net, w);
