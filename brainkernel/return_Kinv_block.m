function [Kinv11, Kinv21] = return_Kinv_block(len_k,xgrid,pid,npid,large_flag,sp_flag,max_unit)
[nx,nc] = size(xgrid);
np = length(pid);
% npid = 1:nx; npid(pid) = [];
nnp = length(npid);
nn = 1e-6;
len_k = exp(-len_k/2);
rho_k = 1;
minl = 1;
if len_k<minl
    len_k = minl;
end

if ~sp_flag
    Kfprior = covSEiso([log(len_k);log(rho_k)],xgrid); % the covariance
    [uu,ss] = svd(Kfprior);
    ssinv = diag(1./(diag(ss)+nn));
    KpriorInv = uu*ssinv*uu';
    Kinv21 = KpriorInv(npid,pid);
    Kinv11 = KpriorInv(pid,pid);
else
    xgridcell = mat2tiles(xgrid,[nx,1]);
    [Bmats,Cdiag] = myfft_nu(xgridcell,len_k,rho_k^(2/nc),len_k,len_k,large_flag);
    BB = column_kron(Bmats);
    %     K = BB'*diag(Cdiag)*BB;
    
    BB11 = BB(:,pid);
    BB22 = BB(:,npid);
    
    %     BB11pinv = (BB11*BB11'+nn*eye(size(BB11,1)))\BB11;
    %     BB22pinv = (BB22*BB22'+nn*eye(size(BB22,1)))\BB22;
    %
    %     Cdiagnn = Cdiag+nn;
    %     Cdiaginv = 1./Cdiagnn;
    %     dd = Cdiagnn-Cdiag.*Cdiaginv.*Cdiag;
    %     dd2 = Cdiag.*Cdiaginv./dd;
    %
    %     Kinv11 = BB11pinv'*bsxfun(@times,1./dd,BB11pinv);
    %     Kinv21 = -BB22pinv'*bsxfun(@times,dd2,BB11pinv);
    
    CB = bsxfun(@times,Cdiag,BB11);
    gamma = BB11'*CB;
    beta = CB'*BB22;
    
    if nx<max_unit
        C = BB22'*bsxfun(@times,Cdiag,BB22)+nn*eye(nnp);
        Cinvbeta = C\beta';
    else
        nw = length(Cdiag);
        B2 = BB22*BB22';
        CB2 = bsxfun(@times,Cdiag,B2)+nn*eye(nw);
        CB2inv = pinv(CB2);
        CB2invC = bsxfun(@times,Cdiag',CB2inv);
        Cinvbeta = BB22'*CB2invC*BB11;
    end
    
    
    Kinv11 = pinv(gamma+nn*eye(np)-beta*Cinvbeta);
    Kinv21 = -Cinvbeta*Kinv11;
end

% [Kinv11, Kinv21] = return_Kinv_block(len_k,xgrid,pid,frac,1);
% [Kinv11a, Kinv21a] = return_Kinv_block(len_k,xgrid,pid,frac,0);
% subplot(231),imagesc(Kinv11),colorbar
% subplot(232),imagesc(Kinv11a),colorbar
% subplot(233),imagesc(Kinv11-Kinv11a),colorbar
% subplot(234),imagesc(Kinv21),colorbar
% subplot(235),imagesc(Kinv21a),colorbar
% subplot(236),imagesc(Kinv21-Kinv21a),colorbar