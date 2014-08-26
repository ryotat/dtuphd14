m=200;
n=500;

L1=(sqrt(m)+sqrt(n))^2;
L2=(-sqrt(m)+sqrt(n))^2;

ensembles = { @(m,n)randn(m,n), @(m,n)(2*rand(m,n)-1)*sqrt(3) };
titles = { 'Gaussian', 'Uniform' };
figure
for kk=1:length(ensembles)
  % Generate a random matrix
  A=ensembles{kk}(m,n);
  
  % Compute the singular values
  ss=svd(A);
  
  % Compute the Marchenko-Pastur probability density function
  tt=linspace(sqrt(L1),sqrt(L2),100);
  hn=sqrt((L1-tt.^2).*(tt.^2-L2))./(pi*m*tt)*m*min(abs(diff(tt)));

  
  subplot(1,length(ensembles),kk);
  
  % Plot the empirical singular values in the descending order
  bar(ss)
  
  % Plot the Marchenko-Pastur cumulative distribution function
  hold on; plot(cumsum(hn), tt, 'm--', 'linewidth',2)

  grid on;
  set(gca,'fontsize',14)
  xlabel('Order')
  ylabel('Singular values')
  title(sprintf('%s, size=[%d %d]',titles{kk},m,n))
  xlim([0 min(m,n)+10])
end
