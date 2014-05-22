function [cov, covf] = gpcovar(net, x)
%GPCOVAR Calculate the covariance for a Gaussian Process.

% Check arguments for consistency
errstring = consist(net, 'gp', x);
if ~isempty(errstring);
  error(errstring);
end

ndata = size(x, 1);

% Compute prior covariance
if nargout >= 2
  [covp, covf] = gpcovarp(net, x, x);
else
  covp = gpcovarp(net, x, x);
end

% Add output noise variance
cov = covp + (net.min_noise + exp(net.noise))*eye(ndata);

