function n2 = mdndist2(mixparams, t)
%MDNDIST2 Calculates squared distance between centres of Gaussian kernels and data

% Check arguments for consistency
errstring = consist(mixparams, 'mdnmixes');
if ~isempty(errstring)
  error(errstring);
end

ncentres   = mixparams.ncentres;
dim_target = mixparams.dim_target;
ntarget    = size(t, 1);
if ntarget ~= size(mixparams.centres, 1)
  error('Number of targets does not match number of mixtures')
end
if size(t, 2) ~= mixparams.dim_target
  error('Target dimension does not match mixture dimension')
end

% Build t that suits parameters, that is repeat t for each centre
t = kron(ones(1, ncentres), t);

% Do subtraction and square
diff2 = (t - mixparams.centres).^2;

% Reshape and sum each component
diff2 = reshape(diff2', dim_target, (ntarget*ncentres))';
n2 = sum(diff2, 2);

% Calculate the sum of distance, and reshape
% so that we have a distance for each centre per target
n2 = reshape(n2, ncentres, ntarget)';

