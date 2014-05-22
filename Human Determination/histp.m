function h = histp(x, xmin, xmax, nbins)
%HISTP	Histogram estimate of 1-dimensional probability distribution.

ndata = length(x);

bins = linspace(xmin, xmax, nbins);

binwidth = (xmax - xmin)/nbins;

num = hist(x, bins);

num = num/(ndata*binwidth);

h = bar(bins, num, 0.6);

