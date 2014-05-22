function x = gsamp(mu, covar, nsamp)
%GSAMP	Sample from a Gaussian distribution.

d = size(covar, 1);

mu = reshape(mu, 1, d);   % Ensure that mu is a row vector

[evec, eval] = eig(covar);

coeffs = randn(nsamp, d)*sqrt(eval);

x = ones(nsamp, 1)*mu + coeffs*evec';
