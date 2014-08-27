nrep=100;

sigma=1;
d=100;
ns=[10:10:200, 400:200:1000, 2000:2000:10000];
lambda=1e-3;

w0=randn(d,1);

for jj=1:length(ns)
  n=ns(jj);
  X=randn(n,d);
  for ii=1:nrep
    yy=X*w0+sigma*randn(n,1);
    
    C=train_ridge(X, yy, lambda, struct('center', 0));
    
    err(ii,jj)=norm(C.w-w0)^2;
  
  end
  lmdn=lambda/n; Sigma=X'*X/n; S=inv(Sigma+lmdn*eye(d));
  bias2(jj)=lmdn^2*norm(S*w0)^2;
  var(jj)=sigma^2/n*trace(S*Sigma*S);
end

figure,
h=errorbar(ns, mean(err), std(err));
hold on;
h2=plot(ns, bias2,'--', ns, var, '-.','linewidth', 2); 
set(h,'linewidth',2);
arrayfun(@(h,c)set(h, 'color', c{:}), h2',...
         {[0 .5 0], [1 0 0]});
set(gca,'fontsize',16,'xscale','log','yscale','log');
xlabel('Number of samples n');
ylabel('Estimation error ||w-w*||^2');
title(sprintf('Ridge Regression: number of variables=%d, lambda=%g',d,lambda));
grid on;

ylim([1e-3, 1e+3]);
legend('simulation','bias^2', 'variance');

set(gcf,'position',[122   333   668   473]);
