function net = rbfsetbf(net, options, x)
%RBFSETBF Set basis functions of RBF from data.

errstring = consist(net, 'rbf', x);
if ~isempty(errstring)
  error(errstring);
end

% Create a spherical Gaussian mixture model
mix = gmm(net.nin, net.nhidden, 'spherical');

% Initialise the parameters from the input data
% Just use a small number of k means iterations
kmoptions = zeros(1, 18);
kmoptions(1) = -1;	% Turn off warnings
kmoptions(14) = 5;  % Just 5 iterations to get centres roughly right
mix = gmminit(mix, x, kmoptions);

% Train mixture model using EM algorithm
[mix, options] = gmmem(mix, x, options);

% Now set the centres of the RBF from the centres of the mixture model
net.c = mix.centres;

% options(7) gives scale of function widths
net = rbfsetfw(net, options(7));
