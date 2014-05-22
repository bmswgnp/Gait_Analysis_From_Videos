function net = mdninit(net, prior, t, options)
%MDNINIT Initialise the weights in a Mixture Density Network.

% Initialise network weights from prior: this gives noise around values
% determined later
net.mlp = mlpinit(net.mlp, prior);

if nargin > 2
  % Initialise priors, centres and variances from target data
  temp_mix = gmm(net.mdnmixes.dim_target, net.mdnmixes.ncentres, 'spherical');
  temp_mix = gmminit(temp_mix, t, options);
  
  ncentres = net.mdnmixes.ncentres;
  dim_target = net.mdnmixes.dim_target;

  % Now set parameters in MLP to yield the right values.
  % This involves setting the biases correctly.
  
  % Priors
  net.mlp.b2(1:ncentres) = temp_mix.priors;
  
  % Centres are arranged in mlp such that we have
  % u11, u12, u13, ..., u1c, ... , uj1, uj2, uj3, ..., ujc, ..., um1, uM2, 
  % ..., uMc
  % This is achieved by transposing temp_mix.centres before reshaping
  end_centres = ncentres*(dim_target+1);
  net.mlp.b2(ncentres+1:end_centres) = ...
    reshape(temp_mix.centres', 1, ncentres*dim_target);
  
  % Variances
  net.mlp.b2((end_centres+1):net.mlp.nout) = ...
    log(temp_mix.covars);
end
