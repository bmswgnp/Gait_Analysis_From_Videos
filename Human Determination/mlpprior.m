function prior = mlpprior(nin, nhidden, nout, aw1, ab1, aw2, ab2)
%MLPPRIOR Create Gaussian prior for mlp.

nextra = nhidden + (nhidden + 1)*nout;
nwts = nin*nhidden + nextra;

if size(aw1) == [1,1] 

    indx = [ones(1, nin*nhidden), zeros(1, nextra)]';
  
elseif size(aw1) == [1, nin]
  
    indx = kron(ones(nhidden, 1), eye(nin));
    indx = [indx; zeros(nextra, nin)];

else
  
    error('Parameter aw1 of invalid dimensions');
    
end

extra = zeros(nwts, 3);

mark1 = nin*nhidden;
mark2 = mark1 + nhidden;
extra(mark1 + 1:mark2, 1) = ones(nhidden,1);
mark3 = mark2 + nhidden*nout;
extra(mark2 + 1:mark3, 2) = ones(nhidden*nout,1);
mark4 = mark3 + nout;
extra(mark3 + 1:mark4, 3) = ones(nout,1);

indx = [indx, extra];

prior.index = indx;
prior.alpha = [aw1, ab1, aw2, ab2]';
