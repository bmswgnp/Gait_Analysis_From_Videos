function [mask, prior] = rbfprior(rbfunc, nin, nhidden, nout, aw2, ab2)
%RBFPRIOR Create Gaussian prior and output layer mask for RBF.

nwts_layer2 = nout + (nhidden *nout);
switch rbfunc
case 'gaussian'
   nwts_layer1 = nin*nhidden + nhidden;
case {'tps', 'r4logr'}
   nwts_layer1 = nin*nhidden;
otherwise
   error('Undefined activation function');
end  
nwts = nwts_layer1 + nwts_layer2;

% Make a mask only for output layer
mask = [zeros(nwts_layer1, 1); ones(nwts_layer2, 1)];

if nargout > 1
  % Construct prior
  indx = zeros(nwts, 2);
  mark2 = nwts_layer1 + (nhidden * nout);
  indx(nwts_layer1 + 1:mark2, 1) = ones(nhidden * nout, 1);
  indx(mark2 + 1:nwts, 2) = ones(nout, 1);

  prior.index = indx;
  prior.alpha = [aw2, ab2]';
end
