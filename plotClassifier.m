function h=plotClassifier(C, xl, yl)

if ~exist('xl','var') || isempty(xl)
  xl=xlim;
end

if ~exist('yl','var') || isempty(yl)
  yl=ylim;
end


col=get(gca,'colororder');
for kk=1:length(C)
  cc=col(mod1(kk,size(col,1)),:);
  w=C(kk).w;
  bias=C(kk).bias;
  if abs(w(1))>abs(w(2))
    h(kk)=plot(-(w(2)*yl+bias)/w(1), yl, '--', 'col', cc, 'linewidth',2);
  else
    h(kk)=plot(xl, -(w(1)*xl+bias)/w(2),'--', 'col', cc, 'linewidth',2);
  end
end
