function g = rosegrad(x)
%ROSEGRAD Calculate gradient of Rosenbrock's function.
%
%	Description
%	G = ROSEGRAD(X) computes the gradient of Rosenbrock's function at
%	each row of X, which should have two columns.

nrows = size(x, 1);
g = zeros(nrows,2);

g(:,1) = -400 * (x(:,2) - x(:,1).^2) * x(:,1) - 2 * (1 - x(:,1));
g(:,2) = 200 * (x(:,2) - x(:,1).^2);
