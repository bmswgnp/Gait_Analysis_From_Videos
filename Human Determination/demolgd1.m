
% Generate the matrix of inputs x and targets t.

ndata = 20;			% Number of data points.
noise = 0.2;			% Standard deviation of noise distribution.
x = [0:1/(ndata - 1):1]';
randn('state', 42);
rand('state', 42);
t = sin(2*pi*x) + noise*randn(ndata, 1);

clc
disp('This demonstration illustrates the use of the on-line gradient')
disp('descent algorithm to train a Multi-Layer Perceptron network for')
disp('regression problems.  It is intended to illustrate the drawbacks')
disp('of this algorithm compared to more powerful non-linear optimisation')
disp('algorithms, such as conjugate gradients.')
disp(' ')
disp('First we generate the data from a noisy sine function and construct')
disp('the network.')
disp(' ')
disp('Press any key to continue.')
pause
% Set up network parameters.
nin = 1;			% Number of inputs.
nhidden = 3;			% Number of hidden units.
nout = 1;			% Number of outputs.
alpha = 0.01;			% Coefficient of weight-decay prior. 

% Create and initialize network weight vector.
net = mlp(nin, nhidden, nout, 'linear');
% Initialise weights reasonably close to 0
net = mlpinit(net, 10);

% Set up vector of options for the optimiser.
options = foptions;
options(1) = 1;			% This provides display of error values.
options(14) = 20;		% Number of training cycles. 
options(18) = 0.1;		% Learning rate
%options(17) = 0.4;		% Momentum
options(17) = 0.4;		% Momentum
options(5) = 1; 		% Do randomise pattern order
clc
disp('Then we set the options for the training algorithm.')
disp(['In the first phase of training, which lasts for ',...
    num2str(options(14)), ' cycles,'])
disp(['the learning rate is ', num2str(options(18)), ...
    ' and the momentum is ', num2str(options(17)), '.'])
disp('The error values are displayed at the end of each pass through the')
disp('entire pattern set.')
disp(' ')
disp('Press any key to continue.')
pause

% Train using online gradient descent
[net, options] = olgd(net, options, x, t);

% Now allow learning rate to decay and remove momentum
options(2) = 0;
options(3) = 0;
options(17) = 0.4;	% Turn off momentum
options(5) = 1;		% Randomise pattern order
options(6) = 1;		% Set learning rate decay on
options(14) = 200;
options(18) = 0.1;	% Initial learning rate

disp(['In the second phase of training, which lasts for up to ',...
    num2str(options(14)), ' cycles,'])
disp(['the learning rate starts at ', num2str(options(18)), ...
    ', decaying at 1/t and the momentum is ', num2str(options(17)), '.'])
disp(' ')
disp('Press any key to continue.')
pause
[net, options] = olgd(net, options, x, t);

clc
disp('Now we plot the data, underlying function, and network outputs')
disp('on a single graph to compare the results.')
disp(' ')
disp('Press any key to continue.')
pause

% Plot the data, the original function, and the trained network function.
plotvals = [0:0.01:1]';
y = mlpfwd(net, plotvals);
fh1 = figure;
plot(x, t, 'ob')
hold on
axis([0 1 -1.5 1.5])
fplot('sin(2*pi*x)', [0 1], '--g')
plot(plotvals, y, '-r')
legend('data', 'function', 'network');
hold off

disp('Note the very poor fit to the data: this should be compared with')
disp('the results obtained in demmlp1.')
disp(' ')
disp('Press any key to exit.')
pause
close(fh1);
clear all;
