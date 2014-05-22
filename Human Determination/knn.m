function net = knn(nin, nout, k, tr_in, tr_targets)
%KNN	Creates a K-nearest-neighbour classifier.


net.type = 'knn';
net.nin = nin;
net.nout = nout;
net.k = k;
errstring = consist(net, 'knn', tr_in, tr_targets);
if ~isempty(errstring)
  error(errstring);
end
net.tr_in = tr_in; 
net.tr_targets = tr_targets;

