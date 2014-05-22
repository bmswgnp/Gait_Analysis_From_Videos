function [y, l] = knnfwd(net, x)
%KNNFWD	Forward propagation through a K-nearest-neighbour classifier.

errstring = consist(net, 'knn', x);
if ~isempty(errstring)
  error(errstring);
end

ntest = size(x, 1);		              % Number of input vectors.
nclass = size(net.tr_targets, 2);		% Number of classes.

% Compute matrix of squared distances between input vectors from the training 
% and test sets.  The matrix distsq has dimensions (ntrain, ntest).

distsq = dist2(net.tr_in, x);

% Now sort the distances. This generates a matrix kind of the same 
% dimensions as distsq, in which each column gives the indices of the
% elements in the corresponding column of distsq in ascending order.

[vals, kind] = sort(distsq);
y = zeros(ntest, nclass);

for k=1:net.k
  % We now look at the predictions made by the Kth nearest neighbours alone,
  % and represent this as a 1-of-N coded matrix, and then accumulate the 
  % predictions so far.

  y = y + net.tr_targets(kind(k,:),:);

end

if nargout == 2
  % Convert this set of outputs to labels, randomly breaking ties
  [temp, l] = max((y + 0.1*rand(size(y))), [], 2);
end
