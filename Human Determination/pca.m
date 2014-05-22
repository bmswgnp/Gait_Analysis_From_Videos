function [PCcoeff, PCvec] = pca(data, N)
%PCA	Principal Components Analysis

if nargin == 1
   N = size(data, 2);
end

if nargout == 1
   evals_only = logical(1);
else
   evals_only = logical(0);
end

if N ~= round(N) | N < 1 | N > size(data, 2)
   error('Number of PCs must be integer, >0, < dim');
end

% Find the sorted eigenvalues of the data covariance matrix
if evals_only
   PCcoeff = eigdec(cov(data), N);
else
  [PCcoeff, PCvec] = eigdec(cov(data), N);
end

