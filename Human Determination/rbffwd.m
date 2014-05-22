function [a, z, n2] = rbffwd(net, x)
%RBFFWD	Forward propagation through RBF network with linear outputs.

% Check arguments for consistency
errstring = consist(net, 'rbf', x);
if ~isempty(errstring);
  error(errstring);
end

[ndata, data_dim] = size(x);

% Calculate squared norm matrix, of dimension (ndata, ncentres)
n2 = dist2(x, net.c);

% Switch on activation function type
switch net.actfn

  case 'gaussian'	% Gaussian
    % Calculate width factors: net.wi contains squared widths
    wi2 = ones(ndata, 1) * (2 .* net.wi);

    % Now compute the activations
    z = exp(-(n2./wi2));

  case 'tps'		% Thin plate spline
    z = n2.*log(n2+(n2==0));

  case 'r4logr'		% r^4 log r
    z = n2.*n2.*log(n2+(n2==0));

  otherwise
    error('Unknown activation function in rbffwd')
end

a = z*net.w2 + ones(ndata, 1)*net.b2;
