function g = rbfbkp(net, x, z, n2, deltas)
%RBFBKP	Backpropagate gradient of error function for RBF network.

% Evaluate second-layer gradients.
gw2 = z'*deltas;
gb2 = sum(deltas);

% Evaluate hidden unit gradients
delhid = deltas*net.w2';

gc = zeros(net.nhidden, net.nin);
ndata = size(x, 1);
t1 = ones(ndata, 1);
t2 = ones(1, net.nin);
% Switch on activation function type
switch net.actfn
      
case 'gaussian' % Gaussian
   delhid = (delhid.*z);
   % A loop seems essential, so do it with the shortest index vector
   if (net.nin < net.nhidden)
      for i = 1:net.nin
         gc(:,i) = (sum(((x(:,i)*ones(1, net.nhidden)) - ...
            (ones(ndata, 1)*(net.c(:,i)'))).*delhid, 1)./net.wi)';
      end
   else
      for i = 1:net.nhidden
         gc(i,:) = sum((x - (t1*(net.c(i,:)))./net.wi(i)).*(delhid(:,i)*t2), 1);
      end
   end
   gwi = sum((n2.*delhid)./(2.*(ones(ndata, 1)*(net.wi.^2))), 1);
   
case 'tps'	% Thin plate spline activation function
   delhid = delhid.*(1+log(n2+(n2==0)));
   for i = 1:net.nhidden
      gc(i,:) = sum(2.*((t1*(net.c(i,:)) - x)).*(delhid(:,i)*t2), 1);
   end
   % widths are not adjustable in this model
   gwi = [];
case 'r4logr' % r^4 log r activation function
   delhid = delhid.*(n2.*(1+2.*log(n2+(n2==0))));
   for i = 1:net.nhidden
      gc(i,:) = sum(2.*((t1*(net.c(i,:)) - x)).*(delhid(:,i)*t2), 1);
   end
   % widths are not adjustable in this model
   gwi = [];
otherwise
   error('Unknown activation function in rbfgrad')
end
   
g = [gc(:)', gwi, gw2(:)', gb2];
