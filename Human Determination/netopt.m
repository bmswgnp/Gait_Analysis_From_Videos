function [net, options, varargout] = netopt(net, options, x, t, alg);

optstring = [alg, '(''neterr'', w, options, ''netgrad'', net, x, t)'];

% Extract weights from network as single vector
w = netpak(net);

% Carry out optimisation
[s{1:nargout}] = eval(optstring);
w = s{1};

if nargout > 1
  options = s{2};

  % If there are additional arguments, extract them
  nextra = nargout - 2;
  if nextra > 0
    for i = 1:nextra
      varargout{i} = s{i+2};
    end
  end
end

% Pack the weights back into the network
net = netunpak(net, w);
