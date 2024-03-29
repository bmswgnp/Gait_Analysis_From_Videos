
% Generate the data

ndata = 500;

% Fix the seeds for reproducible results
randn('state', 42);
rand('state', 42);
data = randn(ndata, 2);
prior = [0.3 0.5 0.2];
% Mixture model swaps clusters 1 and 3
datap = [0.2 0.5 0.3];
datac = [0 2; 0 0; 2 3.5];
datacov = repmat(eye(2), [1 1 3]);
data1 = data(1:prior(1)*ndata,:);
data2 = data(prior(1)*ndata+1:(prior(2)+prior(1))*ndata, :);
data3 = data((prior(1)+prior(2))*ndata +1:ndata, :);

% First cluster has axis aligned variance and centre (2, 3.5)
data1(:, 1) = data1(:, 1)*0.4 + 2.0;
data1(:, 2) = data1(:, 2)*0.8 + 3.5;
datacov(:, :, 3) = [0.4*0.4 0; 0 0.8*0.8];

% Second cluster has variance axes rotated by 30 degrees and centre (0, 0)
rotn = [cos(pi/6) -sin(pi/6); sin(pi/6) cos(pi/6)];
data2(:,1) = data2(:, 1)*0.5;
data2 = data2*rotn;
datacov(:, :, 2) = rotn' * [0.25 0; 0 1] * rotn;

% Third cluster is at (0,2)
data3 = data3 + repmat([0 2], prior(3)*ndata, 1);

% Put the dataset together again
data = [data1; data2; data3];

clc
disp('This demonstration illustrates the use of a Gaussian mixture model')
disp('with full covariance matrices to approximate the unconditional ')
disp('probability density of data in a two-dimensional space.')
disp('We begin by generating the data from a mixture of three Gaussians and')
disp('plotting it.')
disp(' ')
disp('The first cluster has axis aligned variance and centre (0, 2).')
disp('The second cluster has variance axes rotated by 30 degrees')
disp('and centre (0, 0).  The third cluster has unit variance and centre')
disp('(2, 3.5).')
disp(' ')
disp('Press any key to continue.')
pause

fh1 = figure;
plot(data(:, 1), data(:, 2), 'o')
set(gca, 'Box', 'on')

% Set up mixture model
ncentres = 3;
input_dim = 2;
mix = gmm(input_dim, ncentres, 'full');

% Initialise the model parameters from the data
options = foptions;
options(14) = 5;	% Just use 5 iterations of k-means in initialisation
mix = gmminit(mix, data, options);

% Print out model
clc
disp('The mixture model has three components and full covariance')
disp('matrices.  The model parameters after initialisation using the')
disp('k-means algorithm are as follows')
disp('    Priors        Centres')
disp([mix.priors' mix.centres])
disp('Covariance matrices are')
disp(mix.covars)
disp('Press any key to continue.')
pause

% Set up vector of options for EM trainer
options = zeros(1, 18);
options(1)  = 1;		% Prints out error values.
options(14) = 50;		% Number of iterations.

disp('We now train the model using the EM algorithm for 50 iterations.')
disp(' ')
disp('Press any key to continue.')
pause
[mix, options, errlog] = gmmem(mix, data, options);

% Print out model
disp(' ')
disp('The trained model has priors and centres:')
disp('    Priors        Centres')
disp([mix.priors' mix.centres])
disp('The data generator has priors and centres')
disp('    Priors        Centres')
disp([datap' datac])
disp('Model covariance matrices are')
disp(mix.covars(:, :, 1))
disp(mix.covars(:, :, 2))
disp(mix.covars(:, :, 3))
disp('Data generator covariance matrices are')
disp(datacov(:, :, 1))
disp(datacov(:, :, 2))
disp(datacov(:, :, 3))
disp('Note the close correspondence between these parameters and those')
disp('of the distribution used to generate the data.  The match for')
disp('covariance matrices is not that close, but would be improved with')
disp('more iterations of the training algorithm.')
disp(' ')
disp('Press any key to continue.')
pause

clc
disp('We now plot the density given by the mixture model as a surface plot.')
disp(' ')
disp('Press any key to continue.')
pause

% Plot the result
x = -4.0:0.2:5.0;
y = -4.0:0.2:5.0;
[X, Y] = meshgrid(x,y);
X = X(:);
Y = Y(:);
grid = [X Y];
Z = gmmprob(mix, grid);
Z = reshape(Z, length(x), length(y));
c = mesh(x, y, Z);
hold on
title('Surface plot of probability density')
hold off
drawnow

clc
disp('The final plot shows the centres and widths, given by one standard')
disp('deviation, of the three components of the mixture model.  The axes')
disp('of the ellipses of constant density are shown.')
disp(' ')
disp('Press any key to continue.')
pause

% Try to calculate a sensible position for the second figure, below the first
fig1_pos = get(fh1, 'Position');
fig2_pos = fig1_pos;
fig2_pos(2) = fig2_pos(2) - fig1_pos(4) - 30;
fh2 = figure('Position', fig2_pos);

h3 = plot(data(:, 1), data(:, 2), 'bo');
axis equal;
hold on
title('Plot of data and covariances')
for i = 1:ncentres
  [v,d] = eig(mix.covars(:,:,i));
  for j = 1:2
    % Ensure that eigenvector has unit length
    v(:,j) = v(:,j)/norm(v(:,j));
    start=mix.centres(i,:)-sqrt(d(j,j))*(v(:,j)');
    endpt=mix.centres(i,:)+sqrt(d(j,j))*(v(:,j)');
    linex = [start(1) endpt(1)];
    liney = [start(2) endpt(2)];
    line(linex, liney, 'Color', 'k', 'LineWidth', 3)
  end
  % Plot ellipses of one standard deviation
  theta = 0:0.02:2*pi;
  x = sqrt(d(1,1))*cos(theta);
  y = sqrt(d(2,2))*sin(theta);
  % Rotate ellipse axes
  ellipse = (v*([x; y]))';
  % Adjust centre
  ellipse = ellipse + ones(length(theta), 1)*mix.centres(i,:);
  plot(ellipse(:,1), ellipse(:,2), 'r-');
end
hold off

disp('Note how the data cluster positions and widths are captured by')
disp('the mixture model.')
disp(' ')
disp('Press any key to end.')
pause

close(fh1);
close(fh2);
clear all; 

