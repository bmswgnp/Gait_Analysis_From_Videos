function gmmmixes = mdn2gmm(mdnmixes)
%MDN2GMM Converts an MDN mixture data structure to array of GMMs.

% Check argument for consistency
errstring = consist(mdnmixes, 'mdnmixes');
if ~isempty(errstring)
  error(errstring);
end

nmixes = size(mdnmixes.centres, 1);
% Construct ndata structures containing the mixture model information.
% First allocate the memory.
tempmix = gmm(mdnmixes.dim_target, mdnmixes.ncentres, 'spherical');
f = fieldnames(tempmix);
gmmmixes = cell(size(f, 1), 1, nmixes);
gmmmixes = cell2struct(gmmmixes, f,1);

% Then fill each structure in turn using gmmunpak.  Assume that spherical
% covariance structure is used.
for i = 1:nmixes
  centres = reshape(mdnmixes.centres(i, :), mdnmixes.dim_target, ...
    mdnmixes.ncentres)';
  gmmmixes(i) = gmmunpak(tempmix, [mdnmixes.mixcoeffs(i,:), ...
      centres(:)', mdnmixes.covars(i,:)]);
end

