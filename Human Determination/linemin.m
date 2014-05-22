function [x, options] = linemin(f, pt, dir, fpt, options, ...
	varargin)
%LINEMIN One dimensional minimization.

% Set up the options.
if(options(14))
  niters = options(14);
else
  niters = 100;
end
options(10) = 0; % Initialise count of function evaluations

display = options(1);

% Check function string
f = fcnchk(f, length(varargin));

% Value of golden section (1 + sqrt(5))/2.0
phi = 1.6180339887499;
cphi = 1 - 1/phi;
TOL = sqrt(eps);	% Maximal fractional precision
TINY = 1.0e-10;         % Can't use fractional precision when minimum is at 0

% Bracket the minimum
[br_min, br_mid, br_max, num_evals] = feval('minbrack', 'linef', ...
  0.0, 1.0, fpt, f, pt, dir, varargin{:});
options(10) = options(10) + num_evals;  % Increment number of fn. evals
					% No gradient evals in minbrack

% Use Brent's algorithm to find minimum
% Initialise the points and function values
w = br_mid;   	% Where second from minimum is
v = br_mid;   	% Previous value of w
x = v;   	% Where current minimum is
e = 0.0; 	% Distance moved on step before last
fx = feval('linef', x, f, pt, dir, varargin{:});
options(10) = options(10) + 1;
fv = fx; fw = fx;

for n = 1:niters
  xm = 0.5.*(br_min+br_max);  % Middle of bracket
  % Make sure that tolerance is big enough
  tol1 = TOL * (max(abs(x))) + TINY;
  % Decide termination on absolute precision required by options(2)
  if (max(abs(x - xm)) <= options(2) & br_max-br_min < 4*options(2))
    options(8) = fx;
    return;
  end
  % Check if step before last was big enough to try a parabolic step.
  % Note that this will fail on first iteration, which must be a golden
  % section step.
  if (max(abs(e)) > tol1)
    % Construct a trial parabolic fit through x, v and w
    r = (fx - fv) .* (x - w);
    q = (fx - fw) .* (x - v);
    p = (x - v).*q - (x - w).*r;
    q = 2.0 .* (q - r);
    if (q > 0.0) p = -p; end
    q = abs(q);
    % Test if the parabolic fit is OK
    if (abs(p) >= abs(0.5*q*e) | p <= q*(br_min-x) | p >= q*(br_max-x))
      % No it isn't, so take a golden section step
      if (x >= xm)
        e = br_min-x;
      else
        e = br_max-x;
      end
      d = cphi*e;
    else
      % Yes it is, so take the parabolic step
      e = d;
      d = p/q;
      u = x+d;
      if (u-br_min < 2*tol1 | br_max-u < 2*tol1)
        d = sign(xm-x)*tol1;
      end
    end
  else
    % Step before last not big enough, so take a golden section step
    if (x >= xm)
      e = br_min - x;
    else
      e = br_max - x;
    end
    d = cphi*e;
  end
  % Make sure that step is big enough
  if (abs(d) >= tol1)
    u = x+d;
  else
    u = x + sign(d)*tol1;
  end
  % Evaluate function at u
  fu = feval('linef', u, f, pt, dir, varargin{:});
  options(10) = options(10) + 1;
  % Reorganise bracket
  if (fu <= fx)
    if (u >= x)
      br_min = x;
    else
      br_max = x;
    end
    v = w; w = x; x = u;
    fv = fw; fw = fv; fx = fu;
  else
    if (u < x)
      br_min = u;   
    else
      br_max = u;
    end
    if (fu <= fw | w == x)
      v = w; w = u;
      fv = fw; fw = fu;
    elseif (fu <= fv | v == x | v == w)
      v = u;
      fv = fu;
    end
  end
  if (display == 1)
    fprintf(1, 'Cycle %4d  Error %11.6f\n', n, fx);
  end
end
options(8) = fx;
