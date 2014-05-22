function hp = gppak(net)
%GPPAK	Combines GP hyperparameters into one vector.

% Check arguments for consistency
errstring = consist(net, 'gp');
if ~isempty(errstring);
  error(errstring);
end
hp = [net.bias, net.noise, net.inweights, net.fpar];
