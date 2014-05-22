function net = rbfsetfw(net, scale)
%RBFSETFW Set basis function widths of RBF.

% Set the variances to be the largest squared distance between centres
if strcmp(net.actfn, 'gaussian')
   cdist = dist2(net.c, net.c);
   if scale > 0.0
      % Set variance of basis to be scale times average
      % distance to nearest neighbour
      cdist = cdist + realmax*eye(net.nhidden);
      widths = scale*mean(min(cdist));
   else
      widths = max(max(cdist));
   end
   net.wi = widths * ones(size(net.wi));
end
