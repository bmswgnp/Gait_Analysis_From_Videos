function net = glmunpak(net, w)
%GLMUNPAK Separates weights vector into weight and bias matrices. 

% Check arguments for consistency
errstring = consist(net, 'glm');
if ~errstring
  error(errstring);
end

if net.nwts ~= length(w)
  error('Invalid weight vector length')
end

nin = net.nin;
nout = net.nout;
net.w1 = reshape(w(1:nin*nout), nin, nout);
net.b1 = reshape(w(nin*nout + 1: (nin + 1)*nout), 1, nout);
