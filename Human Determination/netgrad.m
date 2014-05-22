function g = netgrad(w, net, x, t)
%NETGRAD Evaluate network error gradient for generic optimizers

gradstr = [net.type, 'grad'];

net = netunpak(net, w);

g = feval(gradstr, net, x, t);
