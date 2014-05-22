function [e, edata, eprior] = gperr(net, x, t)
%GPERR	Evaluate error function for Gaussian Process.

errstring = consist(net, 'gp', x, t);
if ~isempty(errstring);
  error(errstring);
end

cn = gpcovar(net, x);

edata = 0.5*(sum(log(eig(cn, 'nobalance'))) + t'*inv(cn)*t);

% Evaluate the hyperprior contribution to the error.
% The hyperprior is Gaussian with mean pr_mean and variance
% pr_variance
if isfield(net, 'pr_mean')
  w = gppak(net);
  m = repmat(net.pr_mean, size(w));
  if size(net.pr_mean) == [1 1]
    eprior = 0.5*((w-m)*(w-m)');
    e2 = eprior/net.pr_var;
  else
    wpr = repmat(w, size(net.pr_mean, 1), 1)';
    eprior = 0.5*(((wpr - m').^2).*net.index);
    e2 = (sum(e2, 1))*(1./net.pr_var);
  end
else
  e2 = 0;
  eprior = 0;
end

e = edata + e2;

