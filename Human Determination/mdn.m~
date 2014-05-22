function net = mdn(nin, nhidden, ncentres, dim_target, mix_type, ...
	prior, beta)
%MDN	Creates a Mixture Density Network with specified architecture.

% Currently ignore type argument: reserved for future use
net.type = 'mdn';

% Set up the mixture model part of the structure
% For efficiency we use a specialised data structure in place of GMM
mdnmixes.type = 'mdnmixes';
mdnmixes.ncentres = ncentres;
mdnmixes.dim_target = dim_target;

% This calculation depends on spherical variances
mdnmixes.nparams = ncentres + ncentres*dim_target + ncentres;

% Make the weights in the mdnmixes structure null 
mdnmixes.mixcoeffs = [];
mdnmixes.centres = [];
mdnmixes.covars = [];

% Number of output nodes = number of parameters in mixture model
nout = mdnmixes.nparams;

% Set up the MLP part of the network
if (nargin == 5)
  mlpnet = mlp(nin, nhidden, nout, 'linear');
elseif (nargin == 6)
  mlpnet = mlp(nin, nhidden, nout, 'linear', prior);
elseif (nargin == 7)
  mlpnet = mlp(nin, nhidden, nout, 'linear', prior, beta);
end

% Create descriptor
net.mdnmixes = mdnmixes;
net.mlp = mlpnet;
net.nin = nin;
net.nout = dim_target;
net.nwts = mlpnet.nwts;
