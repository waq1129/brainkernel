function [L,dL,fnew] = mle_negll(pp,input_var,pid,fall,nf,nsevar,xgrid,kernelfun,S11)
% unpack input variable
[nx,nc] = size(xgrid);
np = length(pid);
% npid = 1:nx; npid(pid) = [];

[pf,palpha,phyp,pgphyp,optid] = input_var_unpack(pp,input_var,nc,nf,np);
fnew = reshape(pf,[],nf);
alpha = reshape(palpha,nc,nf);
fmu = xgrid*alpha;
hyp = phyp;

fallnew = fall;
fallnew(pid,:) = fnew;
munew = fallnew+fmu;

rbf_p = hyp(1:2);
[k1,k2] = kernelfun([-rbf_p(1)/2;rbf_p(2)/2], munew(pid,:), munew(pid,:), 0, 1) ;
gamma = k1;
gamma = gamma + nsevar*eye(size(gamma));
gammainv = pinv(gamma);

trm1 = .5*logdetns(gamma);

trm2 = .5*trace(S11*gammainv);

L = trm1+trm2;

%% dL_gamma
dL_trm1_gamma = .5*gammainv;
dL_trm2_gamma = -.5*gammainv*S11*gammainv;

dL_gamma = dL_trm1_gamma+dL_trm2_gamma;

df_last = k2;
df_last_gamma = df_last;

dL_u1 = repmat(dL_gamma,nf,1);
dfu = reshape(df_last_gamma,[],np);
duf = dL_u1.*dfu;
duf = sum(reshape(duf,np,[]),1);
dL_uv = 2*reshape(duf,nf,[])';

dL_f = vec(dL_uv);

%% dL_alpha
dL_alpha = zeros(size(palpha));

%
dL_hyp = [0;0;0];

%
dL_gphyp = 0;

%
dL = input_var_pack(dL_f,dL_alpha,dL_hyp,dL_gphyp,optid);

