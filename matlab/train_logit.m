% train_logit - Train a regularized linear logistic regression classifier
% C=train_logit(X, Y, lambda, varargin)
% 
% Ryota Tomioka 2006
function C=train_logit(X, Y, lambda, varargin)

opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, 'use_fminunc', 0,...
                        'display', 0,...
                        'tol', 1e-4,...
                        'MaxIter', size(X,1)*1000,...
                        'solver','qn',...
                        'C0',[],...
                        'bias',1,...
                        'weights',[]);


[n,d] = size(X);

if size(Y,1)~=n
  error('Sample size mismatch!');
end

info = [];

if isempty(opt.C0)
  if opt.bias
    w0 = zeros(d+1,1);
  else
    w0 = zeros(d,1);
  end
else
  if opt.bias
    w0 = [opt.C0.w; opt.C0.bias];
  else
    w0 = opt.C0.w;
  end
end

if isempty(opt.weights)
  opt.weights = ones(n,1);
end


switch(opt.solver)
  case 'nt'
   if opt.bias
     [ww, fval, gg, exitflag]=newton(@objTrain_LogitLLR, w0, opt, X, ...
                                   Y, lambda, opt.weights);
     bias=ww(end);
   else
     [ww, fval, gg, exitflag]=newton(@objTrain_LogitLLR_nobias, w0, opt, X, ...
                                   Y, lambda, opt.weights);
     bias=0;
   end     
   info.fval     = fval;
   info.gval = sum(abs(gg));
   info.algorithm = 'newton (within train_RegLLR)';
   info.exitflag = exitflag;
   if info.exitflag<=0
     warning('TRAIN_REGLLR:NONCONVERGENCE', 'fminunc did not converge.');
   end
 
 case 'qn'

  if opt.bias
    [ww, info]=lbfgs(@(w)objTrain_LogitLLR(w,X,Y,lambda,opt.weights), w0,'epsg', opt.tol,'display',opt.display);
    bias=ww(end);
  else
    [ww, info]=lbfgs(@(w)objTrain_LogitLLR_nobias(w,X,Y,lambda,opt.weights), w0,'epsg', opt.tol,'display',opt.display);
    bias=0;
  end
end


C=struct('w',ww(1:d),'bias',bias,'info',info);


function [x, f, g, exitflag] = newton(fun, x0, opt, varargin)

x = x0;
gg = inf;

c = 0;

opt.display = (isnumeric(opt.display) & opt.display~=0) | strcmp(opt.display, 'iter');

if opt.display
  fprintf('---------------------------------------------\n');
end

while gg>opt.tol & c<opt.MaxIter
  [f, g, H] = feval(fun, x, varargin{:});
  x = x -H\g;
  gg = sum(abs(g));

  if opt.display
    fprintf('[%d] f=%g\t%g\n', c, f, gg);
  end
  
  c = c + 1;
end
exitflag = c~=opt.MaxIter;

function varargout= objTrain_LogitLLR(w, X, Y, lambda, weights)
% [f, g, H] = objTrain_LogitLLR(w, X, Y, lambda)
%
% l2-regularized LLR
%
% Ryota Tomioka 2006
[n,d]=size(X);

bias=w(end);
w=w(1:end-1);


out = X*w+bias; Yout=Y.*out;
f = sum(log(1+exp(-Yout)).*weights) + 0.5*lambda*norm(w)^2;
p = 1./(1+exp(Yout)); Ypw = Y.*p.*weights;
g = [-X'*Ypw + lambda*w; -sum(Ypw)];

varargout{1}=f;
varargout{2}=g;

if nargout>2
  H = [X'*diag(p.*(1-p).*weights)*X + lambda*eye(d), X'*(p.*(1-p).*weights);...
       (p.*(1-p).*weights)'*X, sum(p.*(1-p).*weights)];
  varargout{3}=H;
end

% gap = f + sum(p.*(1-p))+0.5*norm(X'*(Y.*p))^2;



function varargout= objTrain_LogitLLR_nobias(w, X, Y, lambda, weights)
% [f, g, H] = objTrain_LogitLLR(w, X, Y, lambda)
%
% l2-regularized LLR
%
% Ryota Tomioka 2006
[n,d]=size(X);

out = X*w; Yout=Y.*out;
f = sum(log(1+exp(-Yout)).*weights) + 0.5*lambda*norm(w)^2;
p = 1./(1+exp(Yout)); Ypw = Y.*p.*weights;
g = -X'*Ypw + lambda*w;

varargout{1}=f;
varargout{2}=g;

if nargout>2
  H = X'*diag(p.*(1-p).*weights)*X + lambda*eye(d);
  varargout{3}=H;
end





