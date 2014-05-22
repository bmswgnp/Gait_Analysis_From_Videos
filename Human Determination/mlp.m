function net = mlp(nin, nhidden, nout, outfunc, prior, beta)
%MLP	Create a 2-layer feedforward network.

net.type = 'mlp';
net.nin = nin;
net.nhidden = nhidden;
net.nout = nout;
net.nwts = (nin + 1)*nhidden + (nhidden + 1)*nout;

outfns = {'linear', 'logistic', 'softmax'};

if sum(strcmp(outfunc, outfns)) == 0
  error('Undefined output function. Exiting.');
else
  net.outfn = outfunc;
end

if nargin > 4
  if isstruct(prior)
    net.alpha = prior.alpha;
    net.index = prior.index;
  elseif size(prior) == [1 1]
    net.alpha = prior;
  else
    error('prior must be a scalar or a structure');
  end  
end

net.w1 = randn(nin, nhidden)/sqrt(nin + 1);
net.b1 = randn(1, nhidden)/sqrt(nin + 1);
net.w2 = randn(nhidden, nout)/sqrt(nhidden + 1);
net.b2 = randn(1, nout)/sqrt(nhidden + 1);

if nargin == 6
  net.beta = beta;
end
