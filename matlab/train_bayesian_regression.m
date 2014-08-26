function C=train_bayesian_regression(Xtr,Ytr,alpha,sigma2)

[n,d]=size(Xtr);

lambda=alpha*sigma2;
if d<n
  mu=(Xtr'*Xtr+lambda*eye(d))\(Xtr'*Ytr);
else
  mu=Xtr'*((Xtr*Xtr'+lambda*eye(n))\Ytr);
end

C=sigma2*inv(Xtr'*Xtr+lambda*eye(d));

C=struct('mu',mu,'C',C);
