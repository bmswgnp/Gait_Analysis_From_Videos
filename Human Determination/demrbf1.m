
% Generate the matrix of inputs x and targets t.
randn('state', 42);
rand('state', 42);
ndata = 20;			% Number of data points.
noise = 0.2;			% Standard deviation of noise distribution.
x = (linspace(0, 1, ndata))';
t = sin(2*pi*x) + noise*randn(ndata, 1);
mu = mean(x);
sigma = std(x);
tr_in = (x - mu)./(sigma);

clc
disp('This demonstration illustrates the use of a Radial Basis Function')
disp('network for regression problems.  The data is generated from a noisy')
disp('sine function.')
disp(' ')
disp('Press any key to continue.')
pause
% Set up network parameters.
nin = 1;			% Number of inputs.
nhidden = 7;			% Number of hidden units.
nout = 1;			% Number of outputs.

clc
disp('We assess the effect of three different activation functions.')
disp('First we create a network with Gaussian activations.')
disp(' ')
disp('Press any key to continue.')
pause
% Create and initialize network weight and parameter vectors.
net = rbf(nin, nhidden, nout, 'gaussian');

disp('A two-stage training algorithm is used: it uses a small number of')
disp('iterations of EM to position the centres, and then the pseudo-inverse')
disp('of the design matrix to find the second layer weights.')
disp(' ')
disp('Press any key to continue.')
pause
disp('Error values from EM training.')
% Use fast training method
options = foptions;
options(1) = 1;		% Display EM training
options(14) = 10;	% number of iterations of EM
net = rbftrain(net, options, tr_in, t);

disp(' ')
disp('Press any key to continue.')
pause
clc
disp('The second RBF network has thin plate spline activations.')
disp('The same centres are used again, so we just need to calculate')
disp('the second layer weights.')
disp(' ')
disp('Press any key to continue.')
pause
% Create a second RBF with thin plate spline functions
net2 = rbf(nin, nhidden, nout, 'tps');

% Re-use previous centres rather than calling rbftrain again
net2.c = net.c;
[y, act2] = rbffwd(net2, tr_in);

% Solve for new output weights and biases from RBF activations
temp = pinv([act2 ones(ndata, 1)]) * t;
net2.w2 = temp(1:nhidden, :);
net2.b2 = temp(nhidden+1, :);

disp('The third RBF network has r^4 log r activations.')
disp(' ')
disp('Press any key to continue.')
pause
% Create a third RBF with r^4 log r functions
net3 = rbf(nin, nhidden, nout, 'r4logr');

% Overwrite weight vector with parameters from first RBF
net3.c = net.c;
[y, act3] = rbffwd(net3, tr_in);
temp = pinv([act3 ones(ndata, 1)]) * t;
net3.w2 = temp(1:nhidden, :);
net3.b2 = temp(nhidden+1, :);

disp('Now we plot the data, underlying function, and network outputs')
disp('on a single graph to compare the results.')
disp(' ')
disp('Press any key to continue.')
pause
% Plot the data, the original function, and the trained network functions.
plotvals = [x(1):0.01:x(end)]';
inputvals = (plotvals-mu)./sigma;
y = rbffwd(net, inputvals);
y2 = rbffwd(net2, inputvals);
y3 = rbffwd(net3, inputvals);
fh1 = figure;

plot(x, t, 'ob')
hold on
xlabel('Input')
ylabel('Target')
axis([x(1) x(end) -1.5 1.5])
[fx, fy] = fplot('sin(2*pi*x)', [x(1) x(end)]);
plot(fx, fy, '-r', 'LineWidth', 2)
plot(plotvals, y, '--g', 'LineWidth', 2)
plot(plotvals, y2, 'k--', 'LineWidth', 2)
plot(plotvals, y3, '-.c', 'LineWidth', 2)
legend('data', 'function', 'Gaussian RBF', 'Thin plate spline RBF', ...
  'r^4 log r RBF');
hold off

disp('RBF training errors are');
disp(['Gaussian ', num2str(rbferr(net, tr_in, t)), ' TPS ',  ...
num2str(rbferr(net2, tr_in, t)), ' R4logr ', num2str(rbferr(net3, tr_in, t))]);

disp(' ')
disp('Press any key to end.')
pause
close(fh1);
clear all;
