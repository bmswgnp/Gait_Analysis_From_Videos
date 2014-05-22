function [h, hdata] = hbayes(net, hdata) 
%HBAYES	Evaluate Hessian of Bayesian error function for network.

if (isfield(net, 'mask'))
  % Extract relevant entries in Hessian
  nmask_rows = size(find(net.mask), 1);
  hdata = reshape(hdata(logical(net.mask*(net.mask'))), ...
     nmask_rows, nmask_rows);
  nwts = nmask_rows;
else
  nwts = net.nwts;
end
if isfield(net, 'beta')
  h = net.beta*hdata;
else
  h = hdata;
end

if isfield(net, 'alpha')
  if size(net.alpha) == [1 1]
    h = h + net.alpha*eye(nwts);
  else
    if isfield(net, 'mask')
      nindx_cols = size(net.index, 2);
      index = reshape(net.index(logical(repmat(net.mask, ...
         1, nindx_cols))), nmask_rows, nindx_cols);
    else
      index = net.index;
    end
    h = h + diag(index*net.alpha);
  end 
end
