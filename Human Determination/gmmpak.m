function p = gmmpak(mix)
%GMMPAK	Combines all the parameters in a Gaussian mixture model into one vector.

errstring = consist(mix, 'gmm');
if ~errstring
  error(errstring);
end

p = [mix.priors, mix.centres(:)', mix.covars(:)'];
if strcmp(mix.covar_type, 'ppca')
  p = [p, mix.lambda(:)', mix.U(:)'];
end
