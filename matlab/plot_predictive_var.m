function plot_predictive_var(xx, pm, pv, color);

n=length(pm);

up = pm+sqrt(pv);
dn = pm-sqrt(pv);

Y = [up(1:end-1); dn(1:end-1); dn(2:end); up(2:end)];
X = [xx(1:end-1); xx(1:end-1); xx(2:end); xx(2:end)];

patch(X, Y, color, 'edgecolor', 'none');