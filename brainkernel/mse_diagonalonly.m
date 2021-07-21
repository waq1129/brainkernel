function [L,dL,fnew] = mse_diagonalonly(pp,input_var,pid,fall,nf,xgrid,kernelfun,S1)
% unpack input variable
[nx,nc] = size(xgrid);
np = length(pid);
% npid = 1:nx; npid(pid) = [];
[pf,palpha,phyp,pgphyp,optid] = input_var_unpack(pp,input_var,nc,nf,np);
fnew = reshape(pf,[],nf);
alpha = reshape(palpha,nc,nf);
fmu = xgrid*alpha;
hyp = phyp;
fall_new = fall; fall_new(pid,:) = fnew;
munew = fall_new+fmu;
mu_new = munew(pid,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute logdet term (stably, using SVD): - n/2 * log det(Cmat)
rbf_p = hyp(1:2);
[k1,k2] = kernelfun([-rbf_p(1)/2;rbf_p(2)/2], mu_new, mu_new);
delta_fnn = k1;

L = trace((S1-delta_fnn)'*(S1-delta_fnn));

%% dL_eta
dL_nn = 2*(delta_fnn-S1);

df_last_eta = k2;

dL_u1 = repmat(dL_nn,nf,1);
dfu = reshape(df_last_eta,[],np);
duf = dL_u1.*dfu;
duf = sum(reshape(duf,np,[]),1);
dL_uv = 2*reshape(duf,nf,[])';
dL_fl1 = dL_uv;
dL_f = vec(dL_fl1);

%% dL_alpha
dL_alpha = vec(alpha*0);

%
dL_hyp = [0;0;0];

%
dL_gphyp = 0;

%
dL = input_var_pack(dL_f,dL_alpha,dL_hyp,dL_gphyp,optid);

