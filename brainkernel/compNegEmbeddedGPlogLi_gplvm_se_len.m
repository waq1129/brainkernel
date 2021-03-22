function [L,dL] = compNegEmbeddedGPlogLi_gplvm_se_len(pp,input_var,nf,Ycov,nsamp,xgrid,kernelfun,lambda,KpriorInv)
% unpack input variable
[nx,nc] = size(xgrid);
[pf,palpha,phyp,pgphyp,optid] = input_var_unpack(pp,input_var,nc,nf,nx);
f = reshape(pf,[],nf);
gphyp = pgphyp;

len_k = gphyp(1); % marginal variance
rho_k = 0;%gphyp(2);  % length scale
Kfprior = covSEiso([-len_k/2;rho_k/2], xgrid);
[uu,ss] = svd(Kfprior);
ssinv = diag(1./(diag(ss)+1e-4));
KpriorInv = uu*ssinv*uu';
trm3 = trace(f'*KpriorInv*f)/nf-logdet(KpriorInv);

L = trm3;

%%
%
dL_K = -KpriorInv*f*f'*KpriorInv/nf+KpriorInv;
dL_len = -0.5*sum(sum(dL_K.*covSEiso([-len_k/2;rho_k/2], xgrid, 1)));
dL_gphyp = [dL_len];

%
dL = input_var_pack(0,0,0,dL_gphyp,optid);
