function w = mlppak(net)
%MLPPAK	Combines weights and biases into one weights vector.

% Check arguments for consistency
errstring = consist(net, 'mlp');
if ~isempty(errstring);
  error(errstring);
end

w = [net.w1(:)', net.b1, net.w2(:)', net.b2];

