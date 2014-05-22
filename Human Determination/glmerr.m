function [e, edata, eprior, y, a] = glmerr(net, x, t)
%GLMERR	Evaluate error function for generalized linear model.

% Check arguments for consistency
errstring = consist(net, 'glm', x, t);
if ~isempty(errstring);
  error(errstring);
end

[y, a] = glmfwd(net, x);

switch net.outfn

  case 'linear'  	% Linear outputs
    edata = 0.5*sum(sum((y - t).^2));

  case 'logistic'  	% Logistic outputs
    edata = - sum(sum(t.*log(y) + (1 - t).*log(1 - y)));

  case 'softmax'   	% Softmax outputs
    edata = - sum(sum(t.*log(y)));

  otherwise
    error(['Unknown activation function ', net.outfn]);
end

[e, edata, eprior] = errbayes(net, edata);
