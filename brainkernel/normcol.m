function [A,A_normc] = normcol(A)

A_normc = sqrt(sqrt(sum(A.^2,1)));
A = A./repmat(A_normc,size(A,1),1);