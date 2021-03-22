function [K_iix,K_iix1] = cov_block_mult_x(xgrid, x, covfun, groupsize, x1)
if nargin<5
    x1 = [];
end
d = size(xgrid,1);
xgrid_group = mat2tiles([1:d],[1,groupsize]);

K_iix = 0;
K_iix1 = 0;
for ii=1:length(xgrid_group)
    xgridii = xgrid(xgrid_group{ii},:);
    K_ii = covfun(xgrid, xgridii);
    K_iix = K_iix+K_ii*x(xgrid_group{ii},:);
    if ~isempty(x1)
        K_iix1 = K_iix1+K_ii*x1(xgrid_group{ii},:);
    end
end

