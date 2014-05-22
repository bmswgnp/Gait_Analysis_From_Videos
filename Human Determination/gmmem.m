function [mix, options, errlog] = gmmem(mix, x, options)
%GMMEM	EM algorithm for Gaussian mixture model.

% Check that inputs are consistent
errstring = consist(mix, 'gmm', x);
if ~isempty(errstring)
  error(errstring);
end

[ndata, xdim] = size(x);

% Sort out the options
if (options(14))
  niters = options(14);
else
  niters = 100;
end

display = options(1);
store = 0;
if (nargout > 2)
  store = 1;	% Store the error values to return them
  errlog = zeros(1, niters);
end
test = 0;
if options(3) > 0.0
  test = 1;	% Test log likelihood for termination
end

check_covars = 0;
if options(5) >= 1
  if display >= 0
    disp('check_covars is on');
  end
  check_covars = 1;	% Ensure that covariances don't collapse
  MIN_COVAR = eps;	% Minimum singular value of covariance matrix
  init_covars = mix.covars;
end

% Main loop of algorithm
for n = 1:niters
  
  % Calculate posteriors based on old parameters
  [post, act] = gmmpost(mix, x);
  
  % Calculate error value if needed
  if (display | store | test)
    prob = act*(mix.priors)';
    % Error value is negative log likelihood of data
    e = - sum(log(prob));
    if store
      errlog(n) = e;
    end
    if display > 0
      fprintf(1, 'Cycle %4d  Error %11.6f\n', n, e);
    end
    if test
      if (n > 1 & abs(e - eold) < options(3))
        options(8) = e;
        return;
      else
        eold = e;
      end
    end
  end
  
  % Adjust the new estimates for the parameters
  new_pr = sum(post, 1);
  new_c = post' * x;
  
  % Now move new estimates to old parameter vectors
  mix.priors = new_pr ./ ndata;
  
  mix.centres = new_c ./ (new_pr' * ones(1, mix.nin));
  
  switch mix.covar_type
  case 'spherical'
    n2 = dist2(x, mix.centres);
    for j = 1:mix.ncentres
      v(j) = (post(:,j)'*n2(:,j));
    end
    mix.covars = ((v./new_pr))./mix.nin;
    if check_covars
      % Ensure that no covariance is too small
      for j = 1:mix.ncentres
        if mix.covars(j) < MIN_COVAR
          mix.covars(j) = init_covars(j);
        end
      end
    end
  case 'diag'
    for j = 1:mix.ncentres
      diffs = x - (ones(ndata, 1) * mix.centres(j,:));
      mix.covars(j,:) = sum((diffs.*diffs).*(post(:,j)*ones(1, ...
        mix.nin)), 1)./new_pr(j);
    end
    if check_covars
      % Ensure that no covariance is too small
      for j = 1:mix.ncentres
        if min(mix.covars(j,:)) < MIN_COVAR
          mix.covars(j,:) = init_covars(j,:);
        end
      end
    end
  case 'full'
    for j = 1:mix.ncentres
      diffs = x - (ones(ndata, 1) * mix.centres(j,:));
      diffs = diffs.*(sqrt(post(:,j))*ones(1, mix.nin));
      mix.covars(:,:,j) = (diffs'*diffs)/new_pr(j);
    end
    if check_covars
      % Ensure that no covariance is too small
      for j = 1:mix.ncentres
        if min(svd(mix.covars(:,:,j))) < MIN_COVAR
          mix.covars(:,:,j) = init_covars(:,:,j);
        end
      end
    end
  case 'ppca'
    for j = 1:mix.ncentres
      diffs = x - (ones(ndata, 1) * mix.centres(j,:));
      diffs = diffs.*(sqrt(post(:,j))*ones(1, mix.nin));
      [tempcovars, tempU, templambda] = ...
	ppca((diffs'*diffs)/new_pr(j), mix.ppca_dim);
      if length(templambda) ~= mix.ppca_dim
	error('Unable to extract enough components');
      else 
        mix.covars(j) = tempcovars;
        mix.U(:, :, j) = tempU;
        mix.lambda(j, :) = templambda;
      end
    end
    if check_covars
      if mix.covars(j) < MIN_COVAR
        mix.covars(j) = init_covars(j);
      end
    end
    otherwise
      error(['Unknown covariance type ', mix.covar_type]);               
  end
end

options(8) = -sum(log(gmmprob(mix, x)));
if (display >= 0)
  disp('Warning: Maximum number of iterations has been exceeded');
end
  
