function [y, extra, invhess] = netevfwd(w, net, x, t, x_test, invhess)
%NETEVFWD Generic forward propagation with evidence for network

func = [net.type, 'evfwd'];
net = netunpak(net, w);
if nargin == 5
  [y, extra, invhess] = feval(func, net, x, t, x_test);
else
  [y, extra, invhess] = feval(func, net, x, t, x_test, invhess);
end
