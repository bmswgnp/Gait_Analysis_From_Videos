function means = gtmlmean(net, data)
%GTMLMEAN Mean responsibility for data in a GTM.

% Check for consistency
errstring = consist(net, 'gtm', data);
if ~isempty(errstring)
  error(errstring);
end

R = gtmpost(net, data);
means = R*net.X;
