% function [L,dL] = learn_alpha(alpha,xgrid_block,CCblock)
% nc = size(xgrid_block,2);
% nf = length(alpha(:))/nc;
% alpha = reshape(alpha,nc,[]);
% fmu = xgrid_block*alpha;
% np = size(fmu,1);
% K = covSEiso_curv([0,0],fmu);
% L = trace((CCblock-K)*(CCblock-K)');
%
% dL_K = -2*(CCblock-K);
% % [~,dK] = covSEiso_deriv([0,0],fmu,fmu,0,1);
% % dalpha = repmat(dL_K,nf,1).*dK';
% %
% % duf = sum(reshape(dalpha,np,[]),1);
% % dL_uv = 2*reshape(duf,nf,[])';
% %
% % dL = vec(xgrid_block'*dL_uv);
%
%
% %% dL_eta
% nx = size(fmu,1);
% [k1,k2] = covSEiso_curv([0;0], fmu, fmu, 1, 1);
% df_last = -reshape(k2(nx+1:end,:),nx,[]);
% df_last_eta = df_last;
%
% dL_u1 = repmat(dL_K,nf,1);
% dfu = reshape(df_last_eta,[],np);
% duf = dL_u1.*dfu;
% duf = sum(reshape(duf,np,[]),1);
% dL_uv = 2*reshape(duf,nf,[])';
%
% dL = vec(xgrid_block'*dL_uv);


function [L,dL] = learn_alpha(alpha,xgrid_block,CCblock,hyp,nsevar)
nc = size(xgrid_block,2);
nf = length(alpha(:))/nc;
alpha = reshape(alpha,nc,[]);
fmu = xgrid_block*alpha;
np = size(fmu,1);
% K = covSEiso_curv([-hyp(1)/2;hyp(2)/2],fmu);
[K,dK] = covSEiso_deriv([-hyp(1)/2;hyp(2)/2],fmu,fmu,0,1);
K = K + nsevar*eye(np);
L = trace((CCblock-K)*(CCblock-K)');
% Kinv = pinv(K);
% L = trace(CCblock*Kinv)-logdet(Kinv);

dL_K = -2*(CCblock-K);
% dL_K = -Kinv*CCblock*Kinv+Kinv;

dL_u1 = repmat(dL_K,nf,1);
dfu = reshape(dK,[],np);
duf = dL_u1.*dfu;
duf = sum(reshape(duf,np,[]),1);
dL_uv = 2*reshape(duf,nf,[])';

dL = vec(xgrid_block'*dL_uv);

