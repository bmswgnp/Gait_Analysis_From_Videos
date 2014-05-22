function g = glmderiv(net, x)
%GLMDERIV Evaluate derivatives of GLM outputs with respect to weights.

% Check arguments for consistency
errstring = consist(net, 'glm', x);
if ~isempty(errstring)
    error(errstring);
end

ndata = size(x, 1);
if isfield(net, 'mask')
  nwts = size(find(net.mask), 1);
  temp = zeros(1, net.nwts);
else
  nwts = net.nwts;
end
g = zeros(ndata, nwts, net.nout);

temp = zeros(net.nwts, net.nout);
for n = 1:ndata
    % Weight matrix w1
    temp(1:(net.nin*net.nout), :) = kron(eye(net.nout), (x(n, :))');
    % Bias term b1
    temp(net.nin*net.nout+1:end, :) = eye(net.nout);
    if isfield(net, 'mask')
	g(n, :, :) = temp(logical(net.mask));
    else
	g(n, :, :) = temp;
    end
end
