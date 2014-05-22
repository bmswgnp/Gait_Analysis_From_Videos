function [y, a] = glmfwd(net, x)
%GLMFWD	Forward propagation through generalized linear model.

% Check arguments for consistency
errstring = consist(net, 'glm', x);
if ~isempty(errstring);
  error(errstring);
end

ndata = size(x, 1);

a = x*net.w1 + ones(ndata, 1)*net.b1;

switch net.outfn

  case 'linear'     % Linear outputs
    y = a;

  case 'logistic'   % Logistic outputs
    % Prevent overflow and underflow: use same bounds as glmerr
    % Ensure that log(1-y) is computable: need exp(a) > eps
    maxcut = -log(eps);
    % Ensure that log(y) is computable
    mincut = -log(1/realmin - 1);
    a = min(a, maxcut);
    a = max(a, mincut);
    y = 1./(1 + exp(-a));

  case 'softmax'   	% Softmax outputs
    nout = size(a,2);
    % Prevent overflow and underflow: use same bounds as glmerr
    % Ensure that sum(exp(a), 2) does not overflow
    maxcut = log(realmax) - log(nout);
    % Ensure that exp(a) > 0
    mincut = log(realmin);
    a = min(a, maxcut);
    a = max(a, mincut);
    temp = exp(a);
    y = temp./(sum(temp, 2)*ones(1,nout));
    % Ensure that log(y) is computable
    y(y<realmin) = realmin;

  otherwise
    error(['Unknown activation function ', net.outfn]);
end
