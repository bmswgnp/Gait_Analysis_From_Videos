function [var, U, lambda] = ppca(x, ppca_dim)
%PPCA	Probabilistic Principal Components Analysis

if ppca_dim ~= round(ppca_dim) | ppca_dim < 1 | ppca_dim > size(x, 2)
   error('Number of PCs must be integer, >0, < dim');
end

[ndata, data_dim] = size(x);
% Assumes that x is centred and responsibility weighted
% covariance matrix
[l Utemp] = eigdec(x, data_dim);
% Zero any negative eigenvalues (caused by rounding)
l(l<0) = 0;
% Now compute the sigma squared values for all possible values
% of q
s2_temp = cumsum(l(end:-1:1))./[1:data_dim]';
% If necessary, reduce the value of q so that var is at least
% eps * largest eigenvalue
q_temp = min([ppca_dim; data_dim-min(find(s2_temp/l(1) > eps))]);
if q_temp ~= ppca_dim
  wstringpart = 'Covariance matrix ill-conditioned: extracted';
  wstring = sprintf('%s %d/%d PCs', ...
      wstringpart, q_temp, ppca_dim);
  warning(wstring);
end
if q_temp == 0
  % All the latent dimensions have disappeared, so we are
  % just left with the noise model
  var = l(1)/data_dim;
  lambda = var*ones(1, ppca_dim);
else
  var = mean(l(q_temp+1:end));
end  
U = Utemp(:, 1:q_temp);
lambda(1:q_temp) = l(1:q_temp);


