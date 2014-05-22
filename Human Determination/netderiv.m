function g = netderiv(w, net, x)
%NETDERIV Evaluate derivatives of network outputs by weights generically.

fstr = [net.type, 'deriv'];
net = netunpak(net, w);
g = feval(fstr, net, x);
