% number of samples
n=20;
ntest=1000;

N=100;

% order of polynomial used in learning
polyorder=5;

% noise standard deviation
sigma=1;

% True function
w0=[1, 0, -1, 0]';
fstar=@(x)polyval(w0,x);
fout=@(z)z+sigma*randn(n, 1);

% loss function
lossfun=@(y,f)(y-f).^2;

% Input distribution
samplex=@(n)bsxfun(@power, randn(n,1), polyorder:-1:0);

% Indices of coefficients to be visualized
ix=[1, 2];

% Hyperparameter: Prior inverse variance
alpha = 1e-5;

% Hyperparameter: Noise variance
sigma2 = sigma^2;

% Test points
xx=bsxfun(@power, (-5:.1:5)', polyorder:-1:0);
Xtest=samplex(ntest);
ytesttrue=fstar(Xtest(:,end-1));

% Demo mode
demo=1;

W=cell(1,N);
err=zeros(1,N);
figure;
kk=1;
xl=[min(xx(:,end-1)), max(xx(:,end-1))]; yl=[-2, 2];
while demo || kk<=N
  km=mod1(kk,N);
  % sample input variables
  X=samplex(n);
  ytrue=fstar(X(:,end-1));

  % sample output variables
  Y=fout(ytrue);

  % Train ridge regression
  C=train_bayesian_regression(X, Y, alpha, sigma2);
  W{km}=C;

  % Compute test error
  err(km)=mean(lossfun(ytesttrue,Xtest*C.mu));
  
  % predictive mean
  pm=xx*C.mu;
  
  % predictive variance
  pv=diag(xx*C.C*xx');
  
  % Plot the samples, learned function, the true function
  subplot(1,2,1);
  if length(get(gca,'children'))>0
    xl=xlim; yl=ylim;
  end
  cla;
  plot_predictive_var(xx(:,end-1)', pm', pv', [.99 .8 .8]);
  hold on;
  h=plot(xx(:,end-1), fstar(xx(:,end-1)), '--', ...
         xx(:,end-1), pm, '-.', ...
         X(:,end-1), Y, 'bx', 'linewidth', 2);
  set(h(1),'color',[.5 .5 .5]);
  xlim(xl); ylim(yl);
  grid on;
  set(gca,'fontsize',14);
  xlabel('Input');
  ylabel('Output');
  title(sprintf('n=%d d=%d', n, polyorder+1));
  legend(h,'true function','learned function', 'samples',...
         'Location', 'SouthEast');
  
  % Plot the estimated coefficients
  ixd=polyorder+1-ix;
  subplot(1,2,2);
  cla;
  hold on;
  for jj=1:min(kk,N)
    c=W{jj};
    plotEllipse(c.mu(ixd), c.C(ixd,ixd), [.5 .5 .5], 1, 6);
  end
  plot(w0(length(w0)-ix(1)), w0(length(w0)-ix(2)), 'm*', 'linewidth', 2);
  plotEllipse(C.mu(ixd), C.C(ixd,ixd), 'b', 2, 6);
%  plotEllipse(mean(W(ixd,1:min(kk,N)),2), cov(W(ixd,1:min(kk,N))'), [0 .5 .5], 1, 6);
  hold off;
  axis equal; grid on;
  set(gca,'fontsize',14);
  xlabel(sprintf('Coefficient for x^%d',ix(1)));
  ylabel(sprintf('Coefficient for x^%d',ix(2)));
  title(sprintf('test err=%g (mean: %g)', err(km), mean(err(1:min(kk,N)))));
  %  title(sprintf('test err=%g %s %g', ...
%                                     mean(err(1:min(kk,N))), char(177), std(err(1:min(kk,N)))), 'fontsize', 16);
  
  if demo
    pause(0.5);
  end
  kk=kk+1;
end
  