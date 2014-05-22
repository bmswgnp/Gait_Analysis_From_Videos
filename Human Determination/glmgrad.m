function [g, gdata, gprior] = glmgrad(net, x, t)
%GLMGRAD Evaluate gradient of error function for generalized linear model.

% Check arguments for consistency
errstring = consist(net, 'glm', x, t);
if ~isempty(errstring);
  error(errstring);
end

y = glmfwd(net, x);
delout = y - t;

gw1 = x'*delout;
gb1 = sum(delout, 1);

gdata = [gw1(:)', gb1];

[g, gdata, gprior] = gbayes(net, gdata);
