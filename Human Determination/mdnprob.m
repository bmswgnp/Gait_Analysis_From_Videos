function [prob,a] = mdnprob(mixparams, t)
%MDNPROB Computes the data probability likelihood for an MDN mixture structure.

% Check arguments for consistency
errstring = consist(mixparams, 'mdnmixes');
if ~isempty(errstring)
  error(errstring);
end

ntarget    = size(t, 1);
if ntarget ~= size(mixparams.centres, 1)
  error('Number of targets does not match number of mixtures')
end
if size(t, 2) ~= mixparams.dim_target
  error('Target dimension does not match mixture dimension')
end

dim_target = mixparams.dim_target;
ntarget    = size(t, 1);

% Calculate squared norm matrix, of dimension (ndata, ncentres)
% vector (ntarget * ncentres)
dist2 = mdndist2(mixparams, t);

% Calculate variance factors
variance = 2.*mixparams.covars;

% Compute the normalisation term
normal  = ((2.*pi).*mixparams.covars).^(dim_target./2);

% Now compute the activations
a = exp(-(dist2./variance))./normal;

% Accumulate negative log likelihood of targets
prob = mixparams.mixcoeffs.*a;
