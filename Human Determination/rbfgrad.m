function [g, gdata, gprior] = rbfgrad(net, x, t)
%RBFGRAD Evaluate gradient of error function for RBF network.

% Check arguments for consistency
switch net.outfn
case 'linear'
   errstring = consist(net, 'rbf', x, t);
case 'neuroscale'
   errstring = consist(net, 'rbf', x);
otherwise
   error(['Unknown output function ', net.outfn]);
end
if ~isempty(errstring);
  error(errstring);
end

ndata = size(x, 1);

[y, z, n2] = rbffwd(net, x);

switch net.outfn
case 'linear'

   % Sum squared error at output units
   delout = y - t;

   gdata = rbfbkp(net, x, z, n2, delout);
   [g, gdata, gprior] = gbayes(net, gdata);

case 'neuroscale'
   % Compute the error gradient with respect to outputs
   y_dist = sqrt(dist2(y, y));
   D = (t - y_dist)./(y_dist+diag(ones(ndata, 1)));
   temp = y';
   gradient = 2.*sum(kron(D, ones(1, net.nout)) .* ...
      (repmat(y, 1, ndata) - repmat((temp(:))', ndata, 1)), 1);
   gradient = (reshape(gradient, net.nout, ndata))';
   % Compute the error gradient
   gdata = rbfbkp(net, x, z, n2, gradient);
   [g, gdata, gprior] = gbayes(net, gdata);
otherwise
   error(['Unknown output function ', net.outfn]);
end

