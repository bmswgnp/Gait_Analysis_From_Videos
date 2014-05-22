
clc;
disp('This demonstration illustrates the application of Bayesian')
disp('re-estimation to determine the hyperparameters in a simple regression')
disp('problem using an RBF netowk. It is based on a the fact that the')
disp('posterior distribution for the output weights of an RBF is Gaussian')
disp('and uses the evidence maximization framework of MacKay.')
disp(' ')
disp('First, we generate a synthetic data set consisting of a single input')
disp('variable x sampled from a Gaussian distribution, and a target variable')
disp('t obtained by evaluating sin(2*pi*x) and adding Gaussian noise.')
disp(' ')
disp('Press any key to see a plot of the data together with the sine function.')
pause;

% Generate the matrix of inputs x and targets t.

ndata = 16;			% Number of data points.
noise = 0.1;			% Standard deviation of noise distribution.
randn('state', 0);
rand('state', 0);
x = 0.25 + 0.07*randn(ndata, 1);
t = sin(2*pi*x) + noise*randn(size(x));

% Plot the data and the original sine function.
h = figure;
nplot = 200;
plotvals = linspace(0, 1, nplot)';
plot(x, t, 'ok')
xlabel('Input')
ylabel('Target')
hold on
axis([0 1 -1.5 1.5])
fplot('sin(2*pi*x)', [0 1], '-g')
legend('data', 'function');

disp(' ')
disp('Press any key to continue')
pause; clc;

disp('Next we create a two-layer MLP network having 3 hidden units and one')
disp('linear output. The model assumes Gaussian target noise governed by an')
disp('inverse variance hyperparmeter beta, and uses a simple Gaussian prior')
disp('distribution governed by an inverse variance hyperparameter alpha.')
disp(' ');
disp('The network weights and the hyperparameters are initialised and then')
disp('the output layer weights are optimized with the scaled conjugate gradient')
disp('algorithm using the SCG function, with the hyperparameters kept')
disp('fixed. After a maximum of 50 iterations, the hyperparameters are')
disp('re-estimated using the EVIDENCE function. The process of optimizing')
disp('the weights with fixed hyperparameters and then re-estimating the')
disp('hyperparameters is repeated for a total of 3 cycles.')
disp(' ')
disp('Press any key to train the network and determine the hyperparameters.')
pause;

% Set up network parameters.
nin = 1;		% Number of inputs.
nhidden = 3;		% Number of hidden units.
nout = 1;		% Number of outputs.
alpha = 0.01;		% Initial prior hyperparameter. 
beta_init = 50.0;	% Initial noise hyperparameter.

% Create and initialize network weight vector.
net = rbf(nin, nhidden, nout, 'tps', 'linear', alpha, beta_init);
[net.mask, prior] = rbfprior('tps', nin, nhidden, nout, alpha, alpha);
net = netinit(net, prior);

options = foptions;
options(14) = 5;  % At most 5 EM iterations for basis functions
options(1) = -1;  % Turn off all messages
net = rbfsetbf(net, options, x);  % Initialise the basis functions

% Now train the network
nouter = 5;
ninner = 2;
options = foptions;
options(1) = 1;
options(2) = 1.0e-5;		% Absolute precision for weights.
options(3) = 1.0e-5;		% Precision for objective function.
options(14) = 50;		% Number of training cycles in inner loop. 

% Train using scaled conjugate gradients, re-estimating alpha and beta.
for k = 1:nouter
  net = netopt(net, options, x, t, 'scg');
  [net, gamma] = evidence(net, x, t, ninner);
  fprintf(1, '\nRe-estimation cycle %d:\n', k);
  fprintf(1, '  alpha =  %8.5f\n', net.alpha);
  fprintf(1, '  beta  =  %8.5f\n', net.beta);
  fprintf(1, '  gamma =  %8.5f\n\n', gamma);
  disp(' ')
  disp('Press any key to continue.')
  pause;
end

fprintf(1, 'true beta: %f\n', 1/(noise*noise));

disp(' ')
disp('Network training and hyperparameter re-estimation are now complete.') 
disp('Compare the final value for the hyperparameter beta with the true') 
disp('value.')
disp(' ')
disp('Notice that the final error value is close to the number of data')
disp(['points (', num2str(ndata),') divided by two.'])
disp(' ')
disp('Press any key to continue.')
pause; clc;
disp('We can now plot the function represented by the trained network. This')
disp('corresponds to the mean of the predictive distribution. We can also')
disp('plot ''error bars'' representing one standard deviation of the')
disp('predictive distribution around the mean.')
disp(' ')
disp('Press any key to add the network function and error bars to the plot.')
pause;

% Evaluate error bars.
[y, sig2] = netevfwd(netpak(net), net, x, t, plotvals);
sig = sqrt(sig2);

% Plot the data, the original function, and the trained network function.
[y, z] = rbffwd(net, plotvals);
figure(h); hold on;
plot(plotvals, y, '-r')
xlabel('Input')
ylabel('Target')
plot(plotvals, y + sig, '-b');
plot(plotvals, y - sig, '-b');
legend('data', 'function', 'network', 'error bars');

disp(' ')
disp('Notice how the confidence interval spanned by the ''error bars'' is')
disp('smaller in the region of input space where the data density is high,')
disp('and becomes larger in regions away from the data.')
disp(' ')
disp('Press any key to end.')
pause; clc; close(h); 

