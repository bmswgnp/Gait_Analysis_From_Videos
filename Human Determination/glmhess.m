function [h, hdata] = glmhess(net, x, t, hdata)
%GLMHESS Evaluate the Hessian matrix for a generalised linear model.

% Check arguments for consistency
errstring = consist(net, 'glm', x, t);
if ~isempty(errstring);
  error(errstring);
end

ndata = size(x, 1);
nparams = net.nwts;
nout = net.nout;
p = glmfwd(net, x);
inputs = [x ones(ndata, 1)];

if nargin == 3
   hdata = zeros(nparams);	% Full Hessian matrix
   % Calculate data component of Hessian
   switch net.outfn

   case 'linear'
      % No weighting function here
      out_hess = [x ones(ndata, 1)]'*[x ones(ndata, 1)];
      for j = 1:nout
         hdata = rearrange_hess(net, j, out_hess, hdata);
      end
   case 'logistic'
      % Each output is independent
      e = ones(1, net.nin+1);
      link_deriv = p.*(1-p);
      out_hess = zeros(net.nin+1);
      for j = 1:nout
         inputs = [x ones(ndata, 1)].*(sqrt(link_deriv(:,j))*e);
         out_hess = inputs'*inputs;   % Hessian for this output
         hdata = rearrange_hess(net, j, out_hess, hdata);
      end
      
   case 'softmax'
      bb_start = nparams - nout + 1;	% Start of bias weights block
      ex_hess = zeros(nparams);	% Contribution to Hessian from single example
      for m = 1:ndata
         X = x(m,:)'*x(m,:);
         a = diag(p(m,:))-((p(m,:)')*p(m,:));
         ex_hess(1:nparams-nout,1:nparams-nout) = kron(a, X);
         ex_hess(bb_start:nparams, bb_start:nparams) = a.*ones(net.nout, net.nout);
         temp = kron(a, x(m,:));
         ex_hess(bb_start:nparams, 1:nparams-nout) = temp;
         ex_hess(1:nparams-nout, bb_start:nparams) = temp';
         hdata = hdata + ex_hess;
      end
    otherwise
      error(['Unknown activation function ', net.outfn]);
    end
end

[h, hdata] = hbayes(net, hdata);

function hdata = rearrange_hess(net, j, out_hess, hdata)

% Because all the biases come after all the input weights,
% we have to rearrange the blocks that make up the network Hessian.
% This function assumes that we are on the jth output and that all outputs
% are independent.

bb_start = net.nwts - net.nout + 1;	% Start of bias weights block
ob_start = 1+(j-1)*net.nin; 	% Start of weight block for jth output
ob_end = j*net.nin;         	% End of weight block for jth output
b_index = bb_start+(j-1);   	% Index of bias weight
% Put input weight block in right place
hdata(ob_start:ob_end, ob_start:ob_end) = out_hess(1:net.nin, 1:net.nin);
% Put second derivative of bias weight in right place
hdata(b_index, b_index) = out_hess(net.nin+1, net.nin+1);
% Put cross terms (input weight v bias weight) in right place
hdata(b_index, ob_start:ob_end) = out_hess(net.nin+1,1:net.nin);
hdata(ob_start:ob_end, b_index) = out_hess(1:net.nin, net.nin+1);

return 
