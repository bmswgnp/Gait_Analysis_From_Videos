function net = gtm(dim_latent, nlatent, dim_data, ncentres, rbfunc, ...
   prior)
%GTM	Create a Generative Topographic Map.

net.type = 'gtm';
% Input to functions is data
net.nin = dim_data;
net.dim_latent = dim_latent;

% Default is no regularisation
if nargin == 5
   prior = 0.0;
end

% Only allow scalar prior
if isstruct(prior) | size(prior) ~= [1 1]
   error('Prior must be a scalar');
end

% Create RBF network
net.rbfnet = rbf(dim_latent, ncentres, dim_data, rbfunc, ...
   'linear', prior);

% Mask all but output weights
net.rbfnet.mask = rbfprior(rbfunc, dim_latent, ncentres, dim_data);

% Create field for GMM output model
net.gmmnet = gmm(dim_data, nlatent, 'spherical');

% Create empty latent data sample
net.X = [];
