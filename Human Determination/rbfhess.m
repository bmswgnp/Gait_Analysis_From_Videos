function [h, hdata] = rbfhess(net, x, t, hdata)
%RBFHESS Evaluate the Hessian matrix for RBF network.

% Check arguments for consistency
errstring = consist(net, 'rbf', x, t);
if ~isempty(errstring);
  error(errstring);
end

if nargin == 3
  % Data term in Hessian needs to be computed
  [a, z] = rbffwd(net, x); 
  hdata = datahess(net, z, t);
end

% Add in effect of regularisation
[h, hdata] = hbayes(net, hdata);

% Sub-function to compute data part of Hessian
function hdata = datahess(net, z, t)

% Only works for output layer Hessian currently
if (isfield(net, 'mask') & ~any(net.mask(...
      1:(net.nwts - net.nout*(net.nhidden+1)))))
  hdata = zeros(net.nwts);
  ndata = size(z, 1);
  out_hess = [z ones(ndata, 1)]'*[z ones(ndata, 1)];
  for j = 1:net.nout
    hdata = rearrange_hess(net, j, out_hess, hdata);
  end
else
  error('Output layer Hessian only.');
end
return

% Sub-function to rearrange Hessian matrix
function hdata = rearrange_hess(net, j, out_hess, hdata)

% Because all the biases come after all the input weights,
% we have to rearrange the blocks that make up the network Hessian.
% This function assumes that we are on the jth output and that all outputs
% are independent.

% Start of bias weights block
bb_start = net.nwts - net.nout + 1;
% Start of weight block for jth output
ob_start = net.nwts - net.nout*(net.nhidden+1) + (j-1)*net.nhidden...
   + 1; 
% End of weight block for jth output
ob_end = ob_start + net.nhidden - 1; 
% Index of bias weight
b_index = bb_start+(j-1);   
% Put input weight block in right place
hdata(ob_start:ob_end, ob_start:ob_end) = out_hess(1:net.nhidden, ...
   1:net.nhidden);
% Put second derivative of bias weight in right place
hdata(b_index, b_index) = out_hess(net.nhidden+1, net.nhidden+1);
% Put cross terms (input weight v bias weight) in right place
hdata(b_index, ob_start:ob_end) = out_hess(net.nhidden+1, ...
   1:net.nhidden);
hdata(ob_start:ob_end, b_index) = out_hess(1:net.nhidden, ...
   net.nhidden+1);

return 
