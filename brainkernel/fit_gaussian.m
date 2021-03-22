function [a,b,fff] = fit_gaussian(xx,yy)

M = [xx ones(size(xx))];
v = (M'*M)\(M'*yy);
a = v(1);
b = v(2);
fff = a*xx+b;


