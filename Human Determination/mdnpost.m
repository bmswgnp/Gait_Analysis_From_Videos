function [post, a] = mdnpost(mixparams, t)
%MDNPOST Computes the posterior probability for each MDN mixture component.

[prob a] = mdnprob(mixparams, t);

s = sum(prob, 2);
% Set any zeros to one before dividing
s = s + (s==0);
post = prob./(s*ones(1, mixparams.ncentres));
