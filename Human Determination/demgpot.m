function g = demgpot(x, mix)

% Computes the potential gradient

temp = (ones(mix.ncentres,1)*x)-mix.centres;
temp = temp.*(gmmactiv(mix,x)'*ones(1, mix.nin));
% Assume spherical covariance structure
if ~strcmp(mix.covar_type, 'spherical')
  error('Spherical covariance only.')
end
temp = temp./(mix.covars'*ones(1, mix.nin));
temp = temp.*(mix.priors'*ones(1, mix.nin));
g = sum(temp, 1)/gmmprob(mix, x);
