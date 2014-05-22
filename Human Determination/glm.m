function net = glm(nin, nout, outfunc, prior, beta)
%GLM	Create a generalized linear model.

net.type = 'glm';
net.nin = nin;
net.nout = nout;
net.nwts = (nin + 1)*nout;

outtfns = {'linear', 'logistic', 'softmax'};

if sum(strcmp(outfunc, outtfns)) == 0
  error('Undefined activation function. Exiting.');
else
  net.outfn = outfunc;
end

if nargin > 3
  if isstruct(prior)
    net.alpha = prior.alpha;
    net.index = prior.index;
  elseif size(prior) == [1 1]
    net.alpha = prior;
  else
    error('prior must be a scalar or structure');
  end
end
  
net.w1 = randn(nin, nout)/sqrt(nin + 1);
net.b1 = randn(1, nout)/sqrt(nin + 1);

if nargin == 5
  net.beta = beta;
end

