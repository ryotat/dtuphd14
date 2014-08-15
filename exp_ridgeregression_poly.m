% number of samples
n=200;
ntest=1000;

N=100;

% loss function
lossfun=@(y,f)(y-f).^2;

% True function
fstar=@(x)x.^3-x.^2-x;

% order of polynomial used in learning
polyorder=3;

% Input distribution
samplex=@(n)bsxfun(@power, randn(n,1), 1:polyorder);

% Indices of coefficients to be visualized
ix=[1,2];

% noise standard deviation
sigma=1;

% Regularization parameter
lmd = 1e-6;

% input variables
X=samplex(n);
ytrue=fstar(X(:,1));

% Test points
xx=bsxfun(@power, (-5.1:5)', 1:polyorder);
Xtest=samplex(ntest);
ytesttrue=fstar(Xtest(:,1));

% Demo mode
demo=1;

W=zeros(polyorder+1, N); % +1 is the bias term
err=zeros(1,N);
figure;
kk=1;
xl=[min(xx(:,1)), max(xx(:,1))]; yl=[min(fstar(xx(:,1)))-1, max(fstar(xx(:,1)))+1];
while demo || kk<=N
  km=mod1(kk,N);
  % sample output variables
  Y=ytrue+sigma*randn(n, 1);

  % Train ridge regression
  C=train_ridge(X, Y, lmd, struct('center', 1));
  W(:,km)=[C.w; C.bias];

  % Compute test error
  err(km)=mean(lossfun(ytesttrue,Xtest*C.w+C.bias));
  
  % Plot the samples, learned function, the true function
  subplot(1,2,1);
  if length(get(gca,'children'))>0
    xl=xlim; yl=ylim;
  end
  plot(X(:,1), Y, 'x', ...
       xx(:,1), xx*C.w+C.bias, '-.', ...
       xx(:,1), fstar(xx(:,1)), 'm--', 'linewidth', 2);
  xlim(xl); ylim(yl);
  grid on;
  set(gca,'fontsize',14);
  xlabel('Input');
  ylabel('Output');
  title(sprintf('n=%d d=%d', n, polyorder+1));
  
  % Plot the estimated coefficients
  subplot(1,2,2);
  P=W(ix,:);
  plot(P(1,1:min(kk,N)), P(2,1:min(kk,N)), 'x', 'color', ...
       [.5 .5 .5], 'linewidth', 2);
  hold on;
  plot(P(1,km), P(2,km), 'x', 'linewidth', 2);
  hold off;
  axis equal; grid on;
  set(gca,'fontsize',14);
  xlabel('Coefficient for x');
  ylabel('Coefficient for x^2');
  title(sprintf('test err=%g (mean: %g)', err(km), mean(err(1:min(kk,N)))));
  %  title(sprintf('test err=%g %s %g', ...
%                                     mean(err(1:min(kk,N))), char(177), std(err(1:min(kk,N)))), 'fontsize', 16);
  
  if demo
    pause(0.5);
  end
  kk=kk+1;
end
  