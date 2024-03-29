
clc;
randn('state', 1729);
rand('state', 1729);
disp('This demonstration illustrates the technique of automatic relevance')
disp('determination (ARD) using a Gaussian Process.')
disp(' ');
disp('First, we set up a synthetic data set involving three input variables:')
disp('x1 is sampled uniformly from the range (0,1) and has a low level of')
disp('added Gaussian noise, x2 is a copy of x1 with a higher level of added')
disp('noise, and x3 is sampled randomly from a Gaussian distribution. The')
disp('single target variable is given by t = sin(2*pi*x1) with additive')
disp('Gaussian noise. Thus x1 is very relevant for determining the target')
disp('value, x2 is of some relevance, while x3 should in principle be')
disp('irrelevant.')
disp(' ');
disp('Press any key to see a plot of t against x1.')
pause;

ndata = 100;
x1 = rand(ndata, 1);
x2 = x1 + 0.05*randn(ndata, 1);
x3 = 0.5 + 0.5*randn(ndata, 1);
x = [x1, x2, x3];
t = sin(2*pi*x1) + 0.1*randn(ndata, 1);

% Plot the data and the original function.
h = figure;
plotvals = linspace(0, 1, 200)';
plot(x1, t, 'ob')
hold on
xlabel('Input x1')
ylabel('Target')
axis([0 1 -1.5 1.5])
[fx, fy] = fplot('sin(2*pi*x)', [0 1]);
plot(fx, fy, '-g', 'LineWidth', 2);
legend('data', 'function');

disp(' ');
disp('Press any key to continue')
pause; clc;

disp('The Gaussian Process has a separate hyperparameter for each input.')
disp('The hyperparameters are trained by error minimisation using the scaled.')
disp('conjugate gradient optimiser.')
disp(' ');
disp('Press any key to create and train the model.')
disp(' ');
pause;

net = gp(3, 'sqexp');
% Initialise the parameters.
prior.pr_mean = 0;
prior.pr_var = 0.1;
net = gpinit(net, x, t, prior);

% Now train to find the hyperparameters.
options = foptions;
options(1) = 1;
options(14) = 30;

[net, options] = netopt(net, options, x, t, 'scg');

rel = exp(net.inweights);

fprintf(1, ...
  '\nFinal hyperparameters:\n\n  bias:\t\t%10.6f\n  noise:\t%10.6f\n', ...
  exp(net.bias), exp(net.noise));
fprintf(1, '  Vertical scale: %8.6f\n', exp(net.fpar(1)));
fprintf(1, '  Input 1:\t%10.6f\n  Input 2:\t%10.6f\n', ...
  rel(1), rel(2));
fprintf(1, '  Input 3:\t%10.6f\n\n', rel(3));
disp(' ');
disp('We see that the inverse lengthscale associated with')
disp('input x1 is large, that of x2 has an intermediate value and the variance')
disp('of weights associated with x3 is small.')
disp(' ');
disp('This implies that the Gaussian Process is giving greatest emphasis')
disp('to x1 and least emphasis to x3, with intermediate emphasis on')
disp('x2 in the covariance function.')
disp(' ')
disp('Since the target t is statistically independent of x3 we might')
disp('expect the weights associated with this input would go to')
disp('zero. However, for any finite data set there may be some chance')
disp('correlation between x3 and t, and so the corresponding hyperparameter remains')
disp('finite.')
disp('Press any key to continue.')
pause

disp('Finally, we plot the output of the Gaussian Process along the line')
disp('x1 = x2 = x3, together with the true underlying function.')
xt = linspace(0, 1, 50);
xtest = [xt', xt', xt'];

cn = gpcovar(net, x);
cninv = inv(cn);
[ytest, sigsq] = gpfwd(net, xtest, cninv);
sig = sqrt(sigsq);

figure(h); hold on;
plot(xt, ytest, '-k');
plot(xt, ytest+(2*sig), '-b', xt, ytest-(2*sig), '-b');
axis([0 1 -1.5 1.5]);
fplot('sin(2*pi*x)', [0 1], '--m');

disp(' ');
disp('Press any key to end.')
pause; clc; close(h); clear all

