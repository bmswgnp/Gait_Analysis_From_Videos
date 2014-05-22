function [x, options, flog, pointlog] = graddesc(f, x, options, gradf, ...
			varargin)
%GRADDESC Gradient descent optimization.

%  Set up the options.
if length(options) < 18
  error('Options vector too short')
end

if (options(14))
  niters = options(14);
else
  niters = 100;
end

line_min_flag = 0; % Flag for line minimisation option
if (round(options(7)) == 1)
  % Use line minimisation
  line_min_flag = 1;
  % Set options for line minimiser
  line_options = foptions;
  if options(15) > 0
    line_options(2) = options(15);
  end
else
  % Learning rate: must be positive
  if (options(18) > 0)
    eta = options(18);
  else
    eta = 0.01;
  end
  % Momentum term: allow zero momentum
  if (options(17) >= 0)
    mu = options(17);
  else
    mu = 0.5;
  end
end

% Check function string
f = fcnchk(f, length(varargin));
gradf = fcnchk(gradf, length(varargin));

% Display information if options(1) > 0
display = options(1) > 0;

% Work out if we need to compute f at each iteration.
% Needed if using line search or if display results or if termination
% criterion requires it.
fcneval = (options(7) | display | options(3));

%  Check gradients
if (options(9) > 0)
  feval('gradchek', x, f, gradf, varargin{:});
end

dxold = zeros(1, size(x, 2));
xold = x;
fold = 0; % Must be initialised so that termination test can be performed
if fcneval
  fnew = feval(f, x, varargin{:});
  options(10) = options(10) + 1;
  fold = fnew;
end

%  Main optimization loop.
for j = 1:niters
  xold = x;
  grad = feval(gradf, x, varargin{:});
  options(11) = options(11) + 1;  % Increment gradient evaluation counter
  if (line_min_flag ~= 1)
    dx = mu*dxold - eta*grad;
    x =  x + dx;
    dxold = dx;
    if fcneval
      fold = fnew;
      fnew = feval(f, x, varargin{:});
      options(10) = options(10) + 1;
    end
  else
    sd = - grad./norm(grad);	% New search direction.
    fold = fnew;
    % Do a line search: normalise search direction to have length 1
    [lmin, line_options] = feval('linemin', f, x, sd, fold, ...
      line_options, varargin{:});
    options(10) = options(10) + line_options(10);
    x = xold + lmin*sd;
    fnew = line_options(8);
  end
  if nargout >= 3
    flog(j) = fnew;
    if nargout >= 4
      pointlog(j, :) = x;
    end
  end
  if display
    fprintf(1, 'Cycle  %5d  Function %11.8f\n', j, fnew);
  end
  if (max(abs(x - xold)) < options(2) & abs(fnew - fold) < options(3))
    % Termination criteria are met
    options(8) = fnew;
    return;
  end
end

if fcneval
  options(8) = fnew;
else
  options(8) = feval(f, x, varargin{:});
  options(10) = options(10) + 1;
end
if (options(1) >= 0)
  disp('Warning: Maximum number of iterations has been exceeded in graddesc');
end
