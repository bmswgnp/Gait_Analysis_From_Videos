function g = mlpbkp(net, x, z, deltas)
%MLPBKP	Backpropagate gradient of error function for 2-layer network.

% Evaluate second-layer gradients.
gw2 = z'*deltas;
gb2 = sum(deltas, 1);

% Now do the backpropagation.
delhid = deltas*net.w2';
delhid = delhid.*(1.0 - z.*z);

% Finally, evaluate the first-layer gradients.
gw1 = x'*delhid;
gb1 = sum(delhid, 1);

g = [gw1(:)', gb1, gw2(:)', gb2];
