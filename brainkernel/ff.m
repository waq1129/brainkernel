function f = ff(x,a,b)
f = a*x+b./x;
f(x<=0) = nan;