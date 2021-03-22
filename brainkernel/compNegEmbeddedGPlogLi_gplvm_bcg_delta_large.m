function [L,dL,fnew] = compNegEmbeddedGPlogLi_gplvm_bcg_delta_large(pp,input_var,pid,fall,nf,Ccov,nsamp,Ainv,xgrid,kernelfun,lambda,Kinv11,Kinv21)
% unpack input variable
[nx,nc] = size(xgrid);
np = length(pid);
[pf,palpha,phyp,pgphyp,optid] = input_var_unpack(pp,input_var,nc,nf,np);
fnew = reshape(pf,[],nf);
alpha = reshape(palpha,nc,nf);
fmu = xgrid*alpha;
hyp = phyp;
f = fall(pid,:);
mu = fall+fmu;
fmu_i = fmu(pid,:);
fallnew = fall; fallnew(pid,:) = fnew;
munew = fallnew+fmu;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute logdet term (stably, using SVD): - n/2 * log det(Cmat)
rbf_p = hyp(1:2);
delta_f1 = kernelfun([-rbf_p(1)/2;rbf_p(2)/2], munew, fnew+fmu_i, 0, 1);
delta_f2 = kernelfun([-rbf_p(1)/2;rbf_p(2)/2], mu, f+fmu_i, 0, 1);
delta_f = delta_f1-delta_f2;
e = speye(nx,nx); e = full(e(:,pid));
delta_f(pid,:) = delta_f(pid,:)/2;

u = [delta_f; e];
u = reshape(u,nx,[]);
v = [e; delta_f];
v = reshape(v,nx,[]);
Ainvu = Ainv*u;
Ainvv = Ainv*v;

vAu = v'*Ainvu+eye(size(u,2));
invvAu = inv(vAu);
logDetvAu = logdetns(vAu);
Ycov = Ccov*nsamp;

% trm1 = .5*nsamp*sum(log(sdiag));
trm12 = .5*nsamp*logDetvAu;
% Compute quadratic term:  -.5 * sum [yi^T Cmat^-1 yi]
trm22 = -.5*trace(invvAu*Ainvv'*Ycov*Ainvu);

% Compute log prior term:  -.5 * (f-mu)'*K^-1*(f-mu)
npid = 1:nx;
npid(pid) = [];
KpriorInvnew = Kinv11;
KpriorInvnewold = Kinv21';
fold = fall(npid,:);
fnew_old = fall(pid,:);
trm32 = .5*trace(fnew'*KpriorInvnew*fnew+2*fnew'*KpriorInvnewold*fold)-.5*trace(fnew_old'*KpriorInvnew*fnew_old+2*fnew_old'*KpriorInvnewold*fold);
L = trm12+trm22+trm32*lambda;

%%
ASA = Ainv*Ycov*Ainv;
dL_u = .5*nsamp*Ainvv*invvAu'-.5*ASA*v*invvAu'+.5*Ainvv*invvAu'*u'*ASA*v*invvAu';
% dL_v = .5*nsamp*Ainvu*invvAu-.5*ASA*u*invvAu+.5*Ainvu*invvAu*v'*ASA*u*invvAu;

[k1,k2] = covSEiso_deriv([-rbf_p(1)/2;rbf_p(2)/2], munew, fnew+fmu_i, 1, 1);
df_last = k2;

dL_u1 = repmat(dL_u(:,1:2:end),nf,1);
dfu = reshape(df_last,[],np);
duf = dL_u1.*dfu;
duf = sum(reshape(duf,nx,[]),1);
dL_uv = 2*reshape(duf,nf,[])';
dL_f = vec(dL_uv)+vec(KpriorInvnew*fnew+KpriorInvnewold*fold)*lambda;

%% dL_alpha
[k1,k2] = covSEiso_deriv([-rbf_p(1)/2;rbf_p(2)/2], munew, fnew+fmu_i, 1, 1);
df_last = k2;
dL_u1 = repmat(dL_u(:,1:2:end),nf,1);
dfu = reshape(df_last,[],np);
duf = dL_u1.*dfu;
duf = sum(reshape(duf,nx,[]),1);
dL_alpha1 = 2*reshape(duf,nf,[])';

df_last = -k2;
df_last_beta = df_last;
df_last_beta(pid,:) = [];
dL_u1 = repmat(dL_u(npid,1:2:end),nf,1);
dfu = reshape(df_last_beta,[],np);
duf = dL_u1.*dfu;
duf = sum(duf,2);
dL_alpha3 = 2*reshape(duf,nx-np,[]);


[k1,k2] = covSEiso_deriv([-rbf_p(1)/2;rbf_p(2)/2], mu, f+fmu_i, 1, 1);
df_last = k2;
dL_u1 = repmat(dL_u(:,1:2:end),nf,1);
dfu = reshape(df_last,[],np);
duf = dL_u1.*dfu;
duf = sum(reshape(duf,nx,[]),1);
dL_alpha2 = -2*reshape(duf,nf,[])';

df_last = -k2;
df_last_beta = df_last;
df_last_beta(pid,:) = [];
dL_u1 = repmat(dL_u(npid,1:2:end),nf,1);
dfu = reshape(df_last_beta,[],np);
duf = dL_u1.*dfu;
duf = sum(duf,2);
dL_alpha4 = -2*reshape(duf,nx-np,[]);

dL_alpha = vec(xgrid(pid,:)'*(dL_alpha1+dL_alpha2))+vec(xgrid(npid,:)'*(dL_alpha3+dL_alpha4));

%
dL_hyp = [0;0;0];

%
dL_gphyp = 0;

%
dL = input_var_pack(dL_f,dL_alpha,dL_hyp,dL_gphyp,optid);


