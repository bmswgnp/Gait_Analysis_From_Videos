function net = somunpak(net, w)
%SOMUNPAK Replaces node weights in SOM.

errstring = consist(net, 'som');
if ~isempty(errstring)
    error(errstring);
end
% Put weights back into network data structure
net.map = reshape(w', [net.nin net.map_size]);
