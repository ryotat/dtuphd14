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

% Regularization parameter
lmd = 1e-5;

% input variables
X=samplex(n);
ytrue=fstar(X(:,end-1));

% Test points
xx=bsxfun(@power, (-5:.1:5)', polyorder:-1:0);
Xtest=samplex(ntest);
ytesttrue=fstar(Xtest(:,end-1));

% Demo mode
demo=1;

W=zeros(polyorder+1, N); % +1 is the bias term
err=zeros(1,N);
figure;
kk=1;
xl=[min(xx(:,end-1)), max(xx(:,end-1))]; yl=[-2, 2];
while demo || kk<=N
  km=mod1(kk,N);
  % sample output variables
  Y=fout(ytrue);

  % Train ridge regression
  C=train_ridge(X, Y, lmd, struct('center', 0));
  W(:,km)=C.w;

  
  % Compute the true error
  pv_true=(fstar(xx(:,end-1))-xx*C.w).^2;

  % Compute predicted test error
  d=polyorder+1; lmdn=lmd/n; Sigma=X'*X/n; S=inv(Sigma+lmdn*eye(d));
  pv=lmd^2*(norm(w0)^2)*sum((xx*S).^2,2)+sigma^2/n*diag(xx*S*Sigma*S*xx');

  % Compute test error
  err(km)=mean(lossfun(ytesttrue,Xtest*C.w+C.bias));
  
  % Plot the samples, learned function, the true function
  if length(get(gca,'children'))>0
    xl=xlim; yl=ylim;
  end
  cla;
  plot_predictive_var(xx(:,end-1)', (xx*C.w)', pv_true', [.8 .8 .99]);
  hold on;
  plot_predictive_var(xx(:,end-1)', (xx*C.w)', pv', [.99 .8 .8]);
  h=plot(xx(:,end-1), fstar(xx(:,end-1)), '--', ...
       xx(:,end-1), xx*C.w, '-.', ...
       X(:,end-1), Y, 'bx', 'linewidth', 2);
  set(h(1),'color',[.5 .5 .5]);
  xlim(xl); ylim(yl);
  grid on;
  set(gca,'fontsize',14);
  xlabel('Input');
  ylabel('Output');
  title(sprintf('n=%d d=%d', n, polyorder+1));
  legend('true error', 'predicted error', 'true function','learned function', 'samples',...
         'Location', 'SouthEast');
  
  if demo
    pause(0.5);
  end
  kk=kk+1;
end
  