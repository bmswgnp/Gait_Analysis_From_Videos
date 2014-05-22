function [post, a] = gtmpost(net, data)
%GTMPOST Latent space responsibility for data in a GTM.

% Check for consistency
errstring = consist(net, 'gtm', data);
if ~isempty(errstring)
  error(errstring);
end

net.gmmnet.centres = rbffwd(net.rbfnet, net.X);

[post, a] = gmmpost(net.gmmnet, data);
