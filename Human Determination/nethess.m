function [h, varargout] = nethess(w, net, x, t, varargin)
%NETHESS Evaluate network Hessian

hess_str = [net.type, 'hess'];

net = netunpak(net, w);

[s{1:nargout}] = feval(hess_str, net, x, t, varargin{:});
h = s{1};
for i = 2:nargout
  varargout{i-1} = s{i};
end
