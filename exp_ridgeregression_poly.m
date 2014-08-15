% number of samples
n=20;

N=100;

% Indices of coefficients to be visualized
ix=[2,3];

% noise standard deviation
sigma=1;

% Regularization parameter
lmd = 1e-6;

% order of polynomial used in learning
polyorder=3;

% True function
fstar=@(x)x.^3-x.^2-x;

% Test points
xx=bsxfun(@power, (-2:0.1:2)', 0:polyorder);

% input variables
X=bsxfun(@power, randn(n, 1), 0:polyorder);
ytrue=fstar(X(:,2));

% True estimator covariance (assuming gaussian inputs)
mus=arrayfun(@(k)(mod(k,2)==0)*prod(1:2:k-1),1:polyorder*2);
H=hankel(0:polyorder, polyorder:polyorder*2); H(1)=2;
XX=arrayfun(@(x)mus(x), H);
Sigmastar = inv(XX+lmd*eye(polyorder+1))*XX*inv(XX+lmd*eye(polyorder+1));
Sigmastar=(Sigmastar+Sigmastar')/2;


W=zeros(polyorder+1, N); % +1 is the bias term
Coeff=cell(1,N);
figure;
kk=1;
xl=[min(xx(:,2)), max(xx(:,2))]; yl=[min(fstar(xx(:,2)))-1, max(fstar(xx(:,2)))+1];
while 1
  km=mod1(kk,N);
  % sample output variables
  Y=ytrue+sigma*randn(n, 1);

  % Train ridge regression
  C=train_ridge(X, Y, lmd, struct('center', 0));
  W(:,km)=C.w;

  % Plot the samples, learned function, the true function
  subplot(1,2,1);
  if length(get(gca,'children'))>0
    xl=xlim; yl=ylim;
  end
  plot(X(:,2), Y, 'x', ...
       xx(:,2), xx*C.w, '-.', ...
       xx(:,2), fstar(xx(:,2)), 'm--', 'linewidth', 2);
  xlim(xl); ylim(yl);
  grid on;
  set(gca,'fontsize',14);
  xlabel('Input');
  ylabel('Output');
  title(sprintf('n=%d d=%d', n, polyorder+1));

  % Plot the estimated coefficients
  subplot(1,2,2); cla;
  P=W(ix,:);
  plot(P(1,1:min(kk,N)), P(2,1:min(kk,N)), 'x', 'color', ...
       [.5 .5 .5], 'linewidth', 2);
  hold on;
  plot(P(1,km), P(2,km), 'x', 'linewidth', 2);
  axis equal; grid on;
  set(gca,'fontsize',14);
  xlabel('Coefficient for x');
  ylabel('Coefficient for x^2');

  pause(0.5);
  kk=kk+1;
end
  