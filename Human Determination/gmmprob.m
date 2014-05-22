function prob = gmmprob(mix, x)
%GMMPROB Computes the data probability for a Gaussian mixture model.

% Check that inputs are consistent
errstring = consist(mix, 'gmm', x);
if ~isempty(errstring)
  error(errstring);
end

% Compute activations
a = gmmactiv(mix, x);

% Form dot product with priors
prob = a * (mix.priors)';
