function [e, edata, eprior] = mlperr(net, x, t)
%MLPERR	Evaluate error function for 2-layer network.

% Check arguments for consistency
errstring = consist(net, 'mlp', x, t);
if ~isempty(errstring);
  error(errstring);
end
[y, z, a] = mlpfwd(net, x);

switch net.outfn

  case 'linear'        % Linear outputs
    edata = 0.5*sum(sum((y - t).^2));

  case 'logistic'      % Logistic outputs
    % Ensure that log(1-y) is computable: need exp(a) > eps
    maxcut = -log(eps);
    % Ensure that log(y) is computable
    mincut = -log(1/realmin - 1);
    a = min(a, maxcut);
    a = max(a, mincut);
    y = 1./(1 + exp(-a));
    edata = - sum(sum(t.*log(y) + (1 - t).*log(1 - y)));

  case 'softmax'       % Softmax outputs
    nout = size(a,2);
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
    edata = - sum(sum(t.*log(y)));

  otherwise
    error(['Unknown activation function ', net.outfn]);  
end
[e, edata, eprior] = errbayes(net, edata);
