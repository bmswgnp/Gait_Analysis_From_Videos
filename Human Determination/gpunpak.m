function net = gpunpak(net, hp)
%GPUNPAK Separates hyperparameter vector into components. 

% Check arguments for consistency
errstring = consist(net, 'gp');
if ~isempty(errstring);
  error(errstring);
end
if net.nwts ~= length(hp)
  error('Invalid weight vector length');
end

net.bias = hp(1);
net.noise = hp(2);

% Unpack input weights
mark1 = 2 + net.nin;
net.inweights = hp(3:mark1);

% Unpack function specific parameters
net.fpar = hp(mark1 + 1:size(hp, 2));

