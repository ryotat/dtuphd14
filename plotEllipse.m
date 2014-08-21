function plotEllipse(mu, Sigma, color, width, msize)
% plotEllipse(mu, Sigma, color, width, msize)

t = 2*pi*(0:.01:1);

[V, D]=eig(Sigma);

X = mu*ones(1,length(t)) + V*sqrt(D)*[cos(t); sin(t)];

plot(X(1,:), X(2,:), 'color', color, 'linewidth', width);

if exist('msize','var') & msize>0
  hold on;
  plot(mu(1), mu(2), 'x', 'color', color, 'markersize', msize);
end
