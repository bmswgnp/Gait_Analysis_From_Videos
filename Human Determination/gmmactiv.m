function a = gmmactiv(mix, x)
%GMMACTIV Computes the activations of a Gaussian mixture model.

% Check that inputs are consistent
errstring = consist(mix, 'gmm', x);
if ~isempty(errstring)
  error(errstring);
end

ndata = size(x, 1);
a = zeros(ndata, mix.ncentres);  % Preallocate matrix

switch mix.covar_type
  
case 'spherical'
  % Calculate squared norm matrix, of dimension (ndata, ncentres)
  n2 = dist2(x, mix.centres);
  
  % Calculate width factors
  wi2 = ones(ndata, 1) * (2 .* mix.covars);
  normal = (pi .* wi2) .^ (mix.nin/2);
  
  % Now compute the activations
  a = exp(-(n2./wi2))./ normal;
  
case 'diag'
  normal = (2*pi)^(mix.nin/2);
  s = prod(sqrt(mix.covars), 2);
  for j = 1:mix.ncentres
    diffs = x - (ones(ndata, 1) * mix.centres(j, :));
    a(:, j) = exp(-0.5*sum((diffs.*diffs)./(ones(ndata, 1) * ...
      mix.covars(j, :)), 2)) ./ (normal*s(j));
  end
  
case 'full'
  normal = (2*pi)^(mix.nin/2);
  for j = 1:mix.ncentres
    diffs = x - (ones(ndata, 1) * mix.centres(j, :));
    % Use Cholesky decomposition of covariance matrix to speed computation
    c = chol(mix.covars(:, :, j));
    temp = diffs/c;
    a(:, j) = exp(-0.5*sum(temp.*temp, 2))./(normal*prod(diag(c)));
  end
case 'ppca'
  log_normal = mix.nin*log(2*pi);
  d2 = zeros(ndata, mix.ncentres);
  logZ = zeros(1, mix.ncentres);
  for i = 1:mix.ncentres
    k = 1 - mix.covars(i)./mix.lambda(i, :);
    logZ(i) = log_normal + mix.nin*log(mix.covars(i)) - ...
      sum(log(1 - k));
    diffs = x - ones(ndata, 1)*mix.centres(i, :);
    proj = diffs*mix.U(:, :, i);
    d2(:,i) = (sum(diffs.*diffs, 2) - ...
      sum((proj.*(ones(ndata, 1)*k)).*proj, 2)) / ...
      mix.covars(i);
  end
  a = exp(-0.5*(d2 + ones(ndata, 1)*logZ));
otherwise
  error(['Unknown covariance type ', mix.covar_type]);
end
  
