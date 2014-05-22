function e = dempot(x, mix)

% Computes the potential (negative log likelihood)
e = -log(gmmprob(mix, x));
