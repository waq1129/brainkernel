function [L,dL,fnew] = compNegEmbeddedGPlogLi_gplvm_bcg_se_large(pp,input_var,pid,fall,nf,xgrid,kernelfun,lambda,Kinv11,Kinv21,S1,S2,nsamp)
% unpack input variable
[nx,nc] = size(xgrid);
np = length(pid);
npid = 1:nx; npid(pid) = [];
[pf,palpha,phyp,pgphyp,optid] = input_var_unpack(pp,input_var,nc,nf,np);
fnew = reshape(pf,[],nf);
alpha = reshape(palpha,nc,nf);
fmu = xgrid*alpha;
hyp = phyp;
fold = fall(npid,:);
fall_new = fall; fall_new(pid,:) = fnew;
munew = fall_new+fmu;
mu_old = munew(npid,:);
mu_new = munew(pid,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute logdet term (stably, using SVD): - n/2 * log det(Cmat)
rbf_p = hyp(1:2);
[k1,k2] = kernelfun([-rbf_p(1)/2;rbf_p(2)/2], munew, mu_new);
delta_fnn = k1(pid,:);
delta_fon = k1(npid,:);

KpriorInvnew = Kinv11;%KpriorInv(pid,pid);
KpriorInvnewold = Kinv21';%KpriorInv(pid,npid);
L = trace((S1-delta_fnn)'*(S1-delta_fnn))+2*trace((S2-delta_fon)'*(S2-delta_fon))...
    +trace(fnew'*KpriorInvnew*fnew+2*fnew'*KpriorInvnewold*fold)/nsamp*lambda;

%% dL_on
dL_on = 4*(delta_fon-S2);
df_last = k2;
df_last_beta = df_last;
df_last_beta(pid,:) = [];
dL_u1 = repmat(dL_on,nf,1);
dfu = reshape(df_last_beta,[],np);
duf = dL_u1.*dfu;
duf = sum(reshape(duf,nx-np,[]),1);
dL_uv = reshape(duf,nf,[])';
if isempty(dL_uv)
    dL_uv = 0;
end
dL_fl = dL_uv;
dL_f = vec(dL_fl)+vec(2*KpriorInvnew*fnew+2*KpriorInvnewold*fold)/nsamp*lambda;

%% dL_eta
dL_nn = 2*(delta_fnn-S1);

df_last_eta = df_last;
df_last_eta(npid,:) = [];

dL_u1 = repmat(dL_nn,nf,1);
dfu = reshape(df_last_eta,[],np);
duf = dL_u1.*dfu;
duf = sum(reshape(duf,np,[]),1);
dL_uv = 2*reshape(duf,nf,[])';
dL_fl1 = dL_uv;
dL_f = vec(dL_fl1)+dL_f;

%% dL_alpha
% df_last = -k2;
% df_last_beta = df_last;
% df_last_beta(pid,:) = [];
% dL_u1 = repmat(dL_on,nf,1);
% dfu = reshape(df_last_beta,[],np);
% duf = dL_u1.*dfu;
% duf = sum(duf,2);
% dL_uv = reshape(duf,nx-np,[]);
% if isempty(dL_uv)
%     dL_uv = 0;
% end
% dL_fl2 = dL_uv;

%
% dL_alpha = vec(xgrid(npid,:)'*dL_fl2)+vec(xgrid(pid,:)'*(dL_fl+dL_fl1));
dL_alpha = vec(alpha*0);

%
dL_hyp = [0;0;0];

%
dL_gphyp = 0;

%
dL = input_var_pack(dL_f,dL_alpha,dL_hyp,dL_gphyp,optid);

