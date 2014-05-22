function [y, sigsq] = gpfwd(net, x, cninv)
%GPFWD	Forward propagation through Gaussian Process.

errstring = consist(net, 'gp', x);
if ~isempty(errstring);
  error(errstring);
end

if ~(isfield(net, 'tr_in') & isfield(net, 'tr_targets'))
   error('Require training inputs and targets');
end

if nargin == 2
  % Inverse covariance matrix not supplied.
  cninv = inv(gpcovar(net, net.tr_in));
end
ktest = gpcovarp(net, x, net.tr_in);

% Predict mean
y = ktest*cninv*net.tr_targets;

if nargout >= 2
  % Predict error bar
  ndata = size(x, 1);
  sigsq = (ones(ndata, 1) * gpcovarp(net, x(1,:), x(1,:))) ...
    - sum((ktest*cninv).*ktest, 2); 
end
