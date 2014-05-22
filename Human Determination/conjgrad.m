function [x, options, flog, pointlog] = conjgrad(f, x, options, gradf, ...
                                    varargin)

%  Set up the options.
if length(options) < 18
  error('Options vector too short')
end

if(options(14))
  niters = options(14);
else
  niters = 100;
end

% Set up options for line search
line_options = foptions;
% Need a precise line search for success
if options(15) > 0
  line_options(2) = options(15);
else
  line_options(2) = 1e-4;
end

display = options(1);

% Next two lines allow conjgrad to work with expression strings
f = fcnchk(f, length(varargin));
gradf = fcnchk(gradf, length(varargin));

%  Check gradients
if (options(9))
  feval('gradchek', x, f, gradf, varargin{:});
end

options(10) = 0;
options(11) = 0;
nparams = length(x);
fnew = feval(f, x, varargin{:});
options(10) = options(10) + 1;
gradnew = feval(gradf, x, varargin{:});
options(11) = options(11) + 1;
d = -gradnew;		% Initial search direction
br_min = 0;
br_max = 1.0;	% Initial value for maximum distance to search along
tol = sqrt(eps);

j = 1;
if nargout >= 3
  flog(j, :) = fnew;
  if nargout == 4
    pointlog(j, :) = x;
  end
end

while (j <= niters)

  xold = x;
  fold = fnew;
  gradold = gradnew;

  gg = gradold*gradold';
  if (gg == 0.0)
    % If the gradient is zero then we are done.
    options(8) = fnew;
    return;
  end

  % This shouldn't occur, but rest of code depends on d being downhill
  if (gradnew*d' > 0)
    d = -d;
    if options(1) >= 0
      warning('search direction uphill in conjgrad');
    end
  end

  line_sd = d./norm(d);
  [lmin, line_options] = feval('linemin', f, xold, line_sd, fold, ...
    line_options, varargin{:});
  options(10) = options(10) + line_options(10);
  options(11) = options(11) + line_options(11);
  % Set x and fnew to be the actual search point we have found
  x = xold + lmin * line_sd;
  fnew = line_options(8);

  % Check for termination
  if (max(abs(x - xold)) < options(2) & max(abs(fnew - fold)) < options(3))
    options(8) = fnew;
    return;
  end

  gradnew = feval(gradf, x, varargin{:});
  options(11) = options(11) + 1;

  % Use Polak-Ribiere formula to update search direction
  gamma = ((gradnew - gradold)*(gradnew)')/gg;
  d = (d .* gamma) - gradnew;

  if (display > 0)
    fprintf(1, 'Cycle %4d  Function %11.6f\n', j, line_options(8));
  end

  j = j + 1;
  if nargout >= 3
    flog(j, :) = fnew;
    if nargout == 4
      pointlog(j, :) = x;
    end
  end
end

% If we get here, then we haven't terminated in the given number of 
% iterations.

options(8) = fold;
if (options(1) >= 0)
  disp('Warning: Maximum number of iterations has been exceeded in conjgrad');
end
