function net = rbf(nin, nhidden, nout, rbfunc, outfunc, prior, beta)
%RBF	Creates an RBF network with specified architecture
%
%	Description
%	NET = RBF(NIN, NHIDDEN, NOUT, RBFUNC) constructs and initialises a
%	radial basis function network returning a data structure NET. The
%	weights are all initialised with a zero mean, unit variance normal
%	distribution, with the exception of the variances, which are set to
%	one. This makes use of the Matlab function RANDN and so the seed for
%	the random weight initialization can be  set using RANDN('STATE', S)
%	where S is the seed value. The activation functions are defined in
%	terms of the distance between the data point and the corresponding
%	centre.  Note that the functions are computed to a convenient
%	constant multiple: for example, the Gaussian is not normalised.
%	(Normalisation is not needed as the function outputs are linearly
%	combined in the next layer.)
%
%	The fields in NET are
%	  type = 'rbf'
%	  nin = number of inputs
%	  nhidden = number of hidden units
%	  nout = number of outputs
%	  nwts = total number of weights and biases
%	  actfn = string defining hidden unit activation function:
%	    'gaussian' for a radially symmetric Gaussian function.
%	    'tps' for r^2 log r, the thin plate spline function.
%	    'r4logr' for r^4 log r.
%	  outfn = string defining output error function:
%	    'linear' for linear outputs (default) and SoS error.
%	    'neuroscale' for Sammon stress measure.
%	  c = centres
%	  wi = squared widths (null for rlogr and tps)
%	  w2 = second layer weight matrix
%	  b2 = second layer bias vector

net.type = 'rbf';
net.nin = nin;
net.nhidden = nhidden;
net.nout = nout;

% Check that function is an allowed type
actfns = {'gaussian', 'tps', 'r4logr'};
outfns = {'linear', 'neuroscale'};
if (strcmp(rbfunc, actfns)) == 0
  error('Undefined activation function.')
else
  net.actfn = rbfunc;
end
if nargin <= 4
   net.outfn = outfns{1};
elseif (strcmp(outfunc, outfns) == 0)
   error('Undefined output function.')
else
   net.outfn = outfunc;
 end

% Assume each function has a centre and a single width parameter, and that
% hidden layer to output weights include a bias.  Only the Gaussian function
% requires a width
net.nwts = nin*nhidden + (nhidden + 1)*nout;
if strcmp(rbfunc, 'gaussian')
  % Extra weights for width parameters
  net.nwts = net.nwts + nhidden;
end

if nargin > 5
  if isstruct(prior)
    net.alpha = prior.alpha;
    net.index = prior.index;
  elseif size(prior) == [1 1]
    net.alpha = prior;
  else
    error('prior must be a scalar or a structure');
  end  
  if nargin > 6
    net.beta = beta;
  end
end

w = randn(1, net.nwts);
net = rbfunpak(net, w);

% Make widths equal to one
if strcmp(rbfunc, 'gaussian')
  net.wi = ones(1, nhidden);
end

if strcmp(net.outfn, 'neuroscale')
  net.mask = rbfprior(rbfunc, nin, nhidden, nout);
end

