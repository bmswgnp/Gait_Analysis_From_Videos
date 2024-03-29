function w = glmpak(net)
%GLMPAK	Combines weights and biases into one weights vector.
%
%	Description
%	W = GLMPAK(NET) takes a network data structure NET and  combines them
%	into a single row vector W.

errstring = consist(net, 'glm');
if ~errstring
  error(errstring);
end

w = [net.w1(:)', net.b1];

