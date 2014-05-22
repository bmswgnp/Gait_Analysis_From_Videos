function [net, options] = glmtrain(net, options, x, t)
%GLMTRAIN Specialised training of generalized linear model

% Check arguments for consistency
errstring = consist(net, 'glm', x, t);
if ~errstring
  error(errstring);
end

if(~options(14))
  options(14) = 100;
end

display = options(1);
% Do we need to test for termination?
test = (options(2) | options(3));

ndata = size(x, 1);
% Add a column of ones for the bias 
inputs = [x ones(ndata, 1)];

% Linear outputs are a special case as they can be found in one step
if strcmp(net.outfn, 'linear')
  if ~isfield(net, 'alpha')
    % Solve for the weights and biases using left matrix divide
    temp = inputs\t;
  elseif size(net.alpha == [1 1])
    % Use normal form equation
    hessian = inputs'*inputs + net.alpha*eye(net.nin+1);
    temp = pinv(hessian)*(inputs'*t);  
  else
    error('Only scalar alpha allowed');
  end
  net.w1 = temp(1:net.nin, :);
  net.b1 = temp(net.nin+1, :);
  % Store error value in options vector
  options(8) = glmerr(net, x, t);
  return;
end

% Otherwise need to use iterative reweighted least squares
e = ones(1, net.nin+1);
for n = 1:options(14)

  switch net.outfn
    case 'logistic'
      if n == 1
        % Initialise model
        p = (t+0.5)/2;
	act = log(p./(1-p));
      end
      link_deriv = p.*(1-p);
      weights = sqrt(link_deriv); % sqrt of weights
      if (min(min(weights)) < eps)
        warning('ill-conditioned weights in glmtrain')
        return
      end
      z = act + (t-p)./link_deriv;
      % Treat each output independently with relevant set of weights
      for j = 1:net.nout
	indep = inputs.*(weights(:,j)*e);
	dep = z(:,j).*weights(:,j);
	temp = indep\dep;
	net.w1(:,j) = temp(1:net.nin);
	net.b1(j) = temp(net.nin+1);
      end
      [err, edata, eprior, p, act] = glmerr(net, x, t);
      if n == 1
        errold = err;
        wold = netpak(net);
      else
        w = netpak(net);
      end
    case 'softmax'
      if n == 1
        % Initialise model: ensure that row sum of p is one no matter
	% how many classes there are
        p = (t + (1/size(t, 2)))/2;
	act = log(p./(1-p));
      end
      if options(5) == 1 | n == 1
        link_deriv = p.*(1-p);
        weights = sqrt(link_deriv); % sqrt of weights
        if (min(min(weights)) < eps)
          warning('ill-conditioned weights in glmtrain')
          return
        end
        z = act + (t-p)./link_deriv;
        % Treat each output independently with relevant set of weights
        for j = 1:net.nout
          indep = inputs.*(weights(:,j)*e);
	  dep = z(:,j).*weights(:,j);
	  temp = indep\dep;
	  net.w1(:,j) = temp(1:net.nin);
	  net.b1(j) = temp(net.nin+1);
        end
        [err, edata, eprior, p, act] = glmerr(net, x, t);
        if n == 1
          errold = err;
          wold = netpak(net);
        else
          w = netpak(net);
        end
      else
	% Exact method of calculation after w first initialised
	% Start by working out Hessian
	Hessian = glmhess(net, x, t);
	temp = p-t;
	gw1 = x'*(temp);
	gb1 = sum(temp, 1);
	gradient = [gw1(:)', gb1];
	% Now compute modification to weights
	deltaw = -gradient*pinv(Hessian);
	w = wold + deltaw;
	net = glmunpak(net, w);
	[err, edata, eprior, p] = glmerr(net, x, t);
    end

    otherwise
      error(['Unknown activation function ', net.outfn]);
   end
   if options(1)
     fprintf(1, 'Cycle %4d Error %11.6f\n', n, err)
   end
   % Test for termination
   % Terminate if error increases
   if err >  errold
     errold = err;
     w = wold;
     options(8) = err;
     fprintf(1, 'Error has increased: terminating\n')
     return;
   end
   if test & n > 1
     if (max(abs(w - wold)) < options(2) & abs(err-errold) < options(3))
       options(8) = err;
       return;
     else
       errold = err;
       wold = w;
     end
   end
end

options(8) = err;
if (options(1) >= 0)
  disp('Warning: Maximum number of iterations has been exceeded');
end
