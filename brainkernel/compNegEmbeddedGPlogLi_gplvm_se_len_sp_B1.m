function [L,dL] = compNegEmbeddedGPlogLi_gplvm_se_len_sp_B1(pp,input_var,nf,xgrid,l_bound,Binvf1,Binvf2,Bf1,Bf2,wwnrm,dL1_B,large_flag)
% unpack input variable
[nx,nc] = size(xgrid);
[~,~,~,pgphyp] = input_var_unpack(pp,input_var,nc,nf,nx);
gphyp = pgphyp;

nn = 1e-4;
len_k = exp(-gphyp(1)/2); % marginal variance
rho_k = 1;%gphyp(2);  % length scale

xposcell = mat2tiles(xgrid,[nx,1]);
[~,Cdiag] = myfft_nu(xposcell,len_k,rho_k,exp(-l_bound(2)/2),exp(-l_bound(1)/2),large_flag);
Cdiagnn = Cdiag+nn;
dd = 1./(4*Cdiag+2*nn);
L = trace(Bf1'*bsxfun(@times,dd,Bf1))...
    +2*trace(Bf2'*bsxfun(@times,dd,Bf1)) ...
    +trace(Bf2'*bsxfun(@times,dd,Bf2)) ...
    +sum(log(Cdiagnn+Cdiag));

%%
const = sqrt(2*pi)^nc*rho_k^nc;
dc_dlen = (const*nc*len_k^(nc-1)*exp(-.5*wwnrm*len_k.^2)-const*len_k^nc*exp(-.5*wwnrm*len_k.^2).*wwnrm*len_k)*len_k*(-0.5);
ddc1 = -dd.^2.*4.*dc_dlen;
ddc2 = 2./(Cdiagnn+Cdiag).*dc_dlen;

dL1 = sum(dL1_B.*ddc1);
dL2 = sum(ddc2);

%
dL = dL1+dL2;%input_var_pack(0,0,0,dL_gphyp,optid);


