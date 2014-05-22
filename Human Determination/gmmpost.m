function [post, a] = gmmpost(mix, x)
%GMMPOST Computes the class posterior probabilities of a Gaussian mixture model.

% Check that inputs are consistent
errstring = consist(mix, 'gmm', x);
if ~isempty(errstring)
  error(errstring);
end

ndata = size(x, 1);

a = gmmactiv(mix, x);

post = (ones(ndata, 1)*mix.priors).*a;
s = sum(post, 2);
% Set any zeros to one before dividing
s = s + (s==0);
post = post./(s*ones(1, mix.ncentres));
