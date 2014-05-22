function [d2, win_nodes] = somfwd(net, x)
%SOMFWD	Forward propagation through a Self-Organising Map.

% Check for consistency
errstring = consist(net, 'som', x);
if ~isempty(errstring)
    error(errstring);
end

% Turn nodes into matrix of centres
nodes = (reshape(net.map, net.nin, net.num_nodes))';
% Compute squared distance matrix
d2 = dist2(x, nodes);
% Find winning node for each pattern: minimum value in each row
[w, win_nodes] = min(d2, [], 2);
