function [r2, d1r2, d2r2, d3r2, d4r2] = r2_xz_matern3(x,z,ell,flag,i1,i2,i3,i4)
%%%%%
% i1, i2, i3, i4 are the partial derivatives at each layer.
eps = 1e-10;
if nargin<=3, flag = 1; end
if nargin<=4, i1 = [1,0]; end
if nargin<=5, i2 = []; end
if nargin<=6, i3 = []; end
if nargin<=7, i4 = []; end
D = size(x,2);

d1 = zeros(2,D); a1 = find(i1~=0); d1(a1,i1(a1)) = 1;
d2 = zeros(2,D); a2 = find(i2~=0); d2(a2,i2(a2)) = 1;
d3 = zeros(2,D); a3 = find(i3~=0); d3(a3,i3(a3)) = 1;
d4 = zeros(2,D); a4 = find(i4~=0); d4(a4,i4(a4)) = 1;

x1 = x/ell;
z1 = z/ell;
if flag
    r2 = sq_dist(x1',z1');
    r2 = r2-min(r2(:))+eps;
else
    r2 = eps;
end
di = bsxfun(@plus, x(:,i1(a1)),-z(:,i1(a1))');
if a1==1
    d1r2 = 2*di/ell^2;
else
    d1r2 = -2*di/ell^2;
end

if a1==a2
    d2r2 = or_op(d1)*or_op(d2)'*(2/ell^2*ones(size(di)));
else
    d2r2 = or_op(d1)*or_op(d2)'*(-2/ell^2*ones(size(di)));
end

d3r2 = 0;%zeros(size(di));
d4r2 = 0;%zeros(size(di));

