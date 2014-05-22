function modes = gtmlmode(net, data)
%GTMLMODE Mode responsibility for data in a GTM.

% Check for consistency
errstring = consist(net, 'gtm', data);
if ~isempty(errstring)
  error(errstring);
end

R = gtmpost(net, data);
% Mode is maximum responsibility
[max_resp, max_index] = max(R, [], 2);
modes = net.X(max_index, :);
