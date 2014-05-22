function mags = gtmmag(net, latent_data)
%GTMMAG	Magnification factors for a GTM

errstring = consist(net, 'gtm');
if ~isempty(errstring)
  error(errstring);
end

Jacs = rbfjacob(net.rbfnet, latent_data);
nlatent = size(latent_data, 1);
mags = zeros(nlatent, 1);
temp = zeros(net.rbfnet.nin, net.rbfnet.nout);
for m = 1:nlatent
  temp = squeeze(Jacs(m, :, :));  % Turn into a 2d matrix
  mags(m) = sqrt(det(temp*temp'));
end
