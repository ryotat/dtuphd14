% number of samples
n=200;
ntest=1000;

N=100;

% Number of dimensions
d=10;

% loss function
lossfun=@(y,f)(1-sign(y.*f))/2;

% True function
w0=[2, -1, zeros(1,d-2)]';
fstar=@(x)x*w0;
C0=struct('w',w0, 'bias', 0);

% Input distribution
samplex=@(n)randn(n,d);

% Indices of coefficients to be visualized
ix=[1,2];

% noise standard deviation
sigma=1;

% Regularization parameter
lmd = 1e-6;

% input variables
X=samplex(n);
ytrue=fstar(X);

% Test points
Xtest=samplex(ntest);
ytesttrue=fstar(Xtest);

% Demo mode
demo=1;

W=zeros(d+1, N); % +1 is the bias term
err=zeros(1,N);
figure;
kk=1;
% xl=[min(xx(:,1)), max(xx(:,1))]; yl=[min(fstar(xx(:,1)))-1, max(fstar(xx(:,1)))+1];
while demo || kk<=N
  km=mod1(kk,N);
  % sample output variables
  pp=exp(ytrue-logsumexp([-ytrue, ytrue],2));
  Y=2*(rand(n,1)>1-pp)-1;

  % Train ridge regression
  C=train_logit(X, Y, lmd);
  W(:,km)=[C.w; C.bias];

  % Compute test error
  err(km)=mean(lossfun(ytesttrue,Xtest*C.w+C.bias));
  
  % Plot the samples, learned function, the true function
  subplot(1,2,1);
%  if length(get(gca,'children'))>0
%    xl=xlim; yl=ylim;
%  end
  h=plotClassifier(C0);
  hold on;
  plot(X(Y>0,ix(1)), X(Y>0,ix(2)), 'rx',...
         X(Y<0,ix(1)), X(Y<0,ix(2)), 'bo',...
         'linewidth', 2);
  plotClassifier(C);
  hold off;
  set(h,'color',[.5 .5 .5]);
%  xlim(xl); ylim(yl);
  grid on;
  set(gca,'fontsize',14);
  xlabel('x1');
  ylabel('x2');
  title(sprintf('n=%d d=%d', n, d+1));
  legend('true function',...
         'positive examples',...
         'negative examples',...
         'learned function', ...
         'Location', 'SouthEast');
  
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
  