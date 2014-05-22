function prob = gtmprob(net, data)
%GTMPROB Probability for data under a GTM.

% Check for consistency
errstring = consist(net, 'gtm', data);
if ~isempty(errstring)
  error(errstring);
end

net.gmmnet.centres = rbffwd(net.rbfnet, net.X);

prob = gmmprob(net.gmmnet, data);
