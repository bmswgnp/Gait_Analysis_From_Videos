function e = mdnerr(net, x, t)
%MDNERR	Evaluate error function for Mixture Density Network.

% Check arguments for consistency
errstring = consist(net, 'mdn', x, t);
if ~isempty(errstring)
  error(errstring);
end

% Get the output mixture models
mixparams = mdnfwd(net, x);

% Compute the probabilities of mixtures
probs     = mdnprob(mixparams, t);
% Compute the error
e       = sum( -log(max(eps, sum(probs, 2))));

