function [gradient, delta] = gradchek(w, func, grad, varargin)
%GRADCHEK Checks a user-defined gradient function using finite differences.

% Reasonable value for step size
epsilon = 1.0e-6;

func = fcnchk(func, length(varargin));
grad = fcnchk(grad, length(varargin));

% Treat
nparams = length(w);
deltaf = zeros(1, nparams);
step = zeros(1, nparams);
for i = 1:nparams
  % Move a small way in the ith coordinate of w
  step(i) = 1.0;
  fplus  = feval('linef', epsilon, func, w, step, varargin{:});
  fminus = feval('linef', -epsilon, func, w, step, varargin{:});
  % Use central difference formula for approximation
  deltaf(i) = 0.5*(fplus - fminus)/epsilon;
  step(i) = 0.0;
end
gradient = feval(grad, w, varargin{:});
fprintf(1, 'Checking gradient ...\n\n');
delta = gradient - deltaf;
fprintf(1, '   analytic   diffs     delta\n\n');
disp([gradient', deltaf', delta'])
