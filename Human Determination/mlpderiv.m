function g = mlpderiv(net, x)
%MLPDERIV Evaluate derivatives of network outputs with respect to weights.

% Check arguments for consistency
errstring = consist(net, 'mlp', x);
if ~isempty(errstring);
  error(errstring);
end

[y, z] = mlpfwd(net, x);

ndata = size(x, 1);

if isfield(net, 'mask')
  nwts = size(find(net.mask), 1);
  temp = zeros(1, net.nwts);
else
  nwts = net.nwts;
end

g = zeros(ndata, nwts, net.nout);
for k = 1 : net.nout
  delta = zeros(1, net.nout);
  delta(1, k) = 1;
  for n = 1 : ndata
    if isfield(net, 'mask')
      temp = mlpbkp(net, x(n, :), z(n, :), delta);
      g(n, :, k) = temp(logical(net.mask));
    else
      g(n, :, k) = mlpbkp(net, x(n, :), z(n, :),...
	delta);
    end
  end
end
