function [net, options, errlog] = gtmem(net, t, options)
%GTMEM	EM algorithm for Generative Topographic Mapping.

% Check that inputs are consistent
errstring = consist(net, 'gtm', t);
if ~isempty(errstring)
  error(errstring);
end

% Sort out the options
if (options(14))
  niters = options(14);
else
  niters = 100;
end

display = options(1);
store = 0;
if (nargout > 2)
  store = 1;	% Store the error values to return them
  errlog = zeros(1, niters);
end
test = 0;
if options(3) > 0.0
  test = 1;	% Test log likelihood for termination
end

% Calculate various quantities that remain constant during training
[ndata, tdim] = size(t);
ND = ndata*tdim;
[net.gmmnet.centres, Phi] = rbffwd(net.rbfnet, net.X);
Phi = [Phi ones(size(net.X, 1), 1)];
PhiT = Phi';
[K, Mplus1] = size(Phi);

A = zeros(Mplus1, Mplus1);
cholDcmp = zeros(Mplus1, Mplus1);
% Use a sparse representation for the weight regularizing matrix.
if (net.rbfnet.alpha > 0)
  Alpha = net.rbfnet.alpha*speye(Mplus1);
  Alpha(Mplus1, Mplus1) = 0;
end 

for n = 1:niters
   % Calculate responsibilities
   [R, act] = gtmpost(net, t);
     % Calculate error value if needed
   if (display | store | test)
      prob = act*(net.gmmnet.priors)';
      % Error value is negative log likelihood of data
      e = - sum(log(max(prob,eps)));
      if store
         errlog(n) = e;
      end
      if display > 0
         fprintf(1, 'Cycle %4d  Error %11.6f\n', n, e);
      end
      if test
         if (n > 1 & abs(e - eold) < options(3))
            options(8) = e;
            return;
         else
            eold = e;
         end
      end
   end

   % Calculate matrix be inverted (Phi'*G*Phi + alpha*I in the papers).
   % Sparse representation of G normally executes faster and saves
   % memory
   if (net.rbfnet.alpha > 0)
      A = full(PhiT*spdiags(sum(R)', 0, K, K)*Phi + ...
         (Alpha.*net.gmmnet.covars(1)));
   else
      A = full(PhiT*spdiags(sum(R)', 0, K, K)*Phi);
   end
   % A is a symmetric matrix likely to be positive definite, so try
   % fast Cholesky decomposition to calculate W, otherwise use SVD.
   % (PhiT*(R*t)) is computed right-to-left, as R
   % and t are normally (much) larger than PhiT.
   [cholDcmp singular] = chol(A);
   if (singular)
      if (display)
         fprintf(1, ...
            'gtmem: Warning -- M-Step matrix singular, using pinv.\n');
      end
      W = pinv(A)*(PhiT*(R'*t));
   else
      W = cholDcmp \ (cholDcmp' \ (PhiT*(R'*t)));
   end
   % Put new weights into network to calculate responsibilities
   % net.rbfnet = netunpak(net.rbfnet, W);
   net.rbfnet.w2 = W(1:net.rbfnet.nhidden, :);
   net.rbfnet.b2 = W(net.rbfnet.nhidden+1, :);
   % Calculate new distances
   d = dist2(t, Phi*W);
   
   % Calculate new value for beta
   net.gmmnet.covars = ones(1, net.gmmnet.ncentres)*(sum(sum(d.*R))/ND);
end

options(8) = -sum(log(gtmprob(net, t)));
if (display >= 0)
  disp('Warning: Maximum number of iterations has been exceeded');
end
