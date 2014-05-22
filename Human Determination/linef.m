function y = linef(lambda, fn, x, d, varargin)
%LINEF	Calculate function value along a line.

% Check function string
fn = fcnchk(fn, length(varargin));

y = feval(fn, x+lambda.*d, varargin{:});
