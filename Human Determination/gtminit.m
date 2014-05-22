function net = gtminit(net, options, data, samp_type, varargin)
%GTMINIT Initialise the weights and latent sample in a GTM.

% Check for consistency
errstring = consist(net, 'gtm', data);
if ~isempty(errstring)
  error(errstring);
end

% Check type of sample
stypes = {'regular', 'uniform', 'gaussian'};
if (strcmp(samp_type, stypes)) == 0
  error('Undefined sample type.')
end

if net.dim_latent > size(data, 2)
  error('Latent space dimension must not be greater than data dimension')
end
nlatent = net.gmmnet.ncentres;
nhidden = net.rbfnet.nhidden;

% Create latent data sample and set RBF centres

switch samp_type
case 'regular'
   if nargin ~= 6
      error('Regular type must specify latent and RBF shapes');
   end
   l_samp_size = varargin{1};
   rbf_samp_size = varargin{2};
   if round(l_samp_size) ~= l_samp_size
      error('Latent sample specification must contain integers')
   end
   % Check existence and size of rbf specification
   if any(size(rbf_samp_size) ~= [1 net.dim_latent]) | ...
         prod(rbf_samp_size) ~= nhidden
      error('Incorrect specification of RBF centres')
   end
   % Check dimension and type of latent data specification
   if any(size(l_samp_size) ~= [1 net.dim_latent]) | ...
         prod(l_samp_size) ~= nlatent
      error('Incorrect dimension of latent sample spec.')
   end
   if net.dim_latent == 1
      net.X = [-1:2/(l_samp_size-1):1]';
      net.rbfnet.c = [-1:2/(rbf_samp_size-1):1]';
      net.rbfnet = rbfsetfw(net.rbfnet, options(7));
   elseif net.dim_latent == 2
      net.X = gtm_rctg(l_samp_size);
      net.rbfnet.c = gtm_rctg(rbf_samp_size);
      net.rbfnet = rbfsetfw(net.rbfnet, options(7));
   else
      error('For regular sample, input dimension must be 1 or 2.')
   end
   
   
case {'uniform', 'gaussian'}
   if strcmp(samp_type, 'uniform')
      net.X = 2 * (rand(nlatent, net.dim_latent) - 0.5);
   else
      % Sample from N(0, 0.25) distribution to ensure most latent 
      % data is inside square
      net.X = randn(nlatent, net.dim_latent)/2;
   end   
   net.rbfnet = rbfsetbf(net.rbfnet, options, net.X);
otherwise
   % Shouldn't get here
   error('Invalid sample type');
   
end

% Latent data sample and basis function parameters chosen.
% Now set output weights
[PCcoeff, PCvec] = pca(data);

% Scale PCs by eigenvalues
A = PCvec(:, 1:net.dim_latent)*diag(sqrt(PCcoeff(1:net.dim_latent)));

[temp, Phi] = rbffwd(net.rbfnet, net.X);
% Normalise X to ensure 1:1 mapping of variances and calculate weights
% as solution of Phi*W = normX*A'
normX = (net.X - ones(size(net.X))*diag(mean(net.X)))*diag(1./std(net.X));
net.rbfnet.w2 = Phi \ (normX*A');
% Bias is mean of target data
net.rbfnet.b2 = mean(data);

% Must also set initial value of variance
% Find average distance between nearest centres
% Ensure that distance of centre to itself is excluded by setting diagonal
% entries to realmax
net.gmmnet.centres = rbffwd(net.rbfnet, net.X);
d = dist2(net.gmmnet.centres, net.gmmnet.centres) + ...
  diag(ones(net.gmmnet.ncentres, 1)*realmax);
sigma = mean(min(d))/2;

% Now set covariance to minimum of this and next largest eigenvalue
if net.dim_latent < size(data, 2)
  sigma = min(sigma, PCcoeff(net.dim_latent+1));
end
net.gmmnet.covars = sigma*ones(1, net.gmmnet.ncentres);

% Sub-function to create the sample data in 2d
function sample = gtm_rctg(samp_size)

xDim = samp_size(1);
yDim = samp_size(2);
% Produce a grid with the right number of rows and columns
[X, Y] = meshgrid([0:1:(xDim-1)], [(yDim-1):-1:0]);

% Change grid representation 
sample = [X(:), Y(:)];

% Shift grid to correct position and scale it
maxXY= max(sample);
sample(:,1) = 2*(sample(:,1) - maxXY(1)/2)./maxXY(1);
sample(:,2) = 2*(sample(:,2) - maxXY(2)/2)./maxXY(2);
return;

   
   
