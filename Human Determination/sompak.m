function [c] = sompak(net)
%SOMPAK	Combines node weights into one weights matrix.

errstring = consist(net, 'som');
if ~isempty(errstring)
    error(errstring);
end
% Returns map as a sequence of row vectors
c = (reshape(net.map, net.nin, net.num_nodes))';
