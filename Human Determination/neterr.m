function [e, varargout] = neterr(w, net, x, t)
%NETERR	Evaluate network error function for generic optimizers

errstr = [net.type, 'err'];
net = netunpak(net, w);

[s{1:nargout}] = feval(errstr, net, x, t);
e = s{1};
if nargout > 1
  for i = 2:nargout
    varargout{i-1} = s{i};
  end
end
