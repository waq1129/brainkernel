function Y1_new = init_LaplaceEigen_sp_Kinv(Y,Y_nsamp,pid,npid,rescale_flag,fmu,lambda,Kinv11,Kinv21)
nx = size(Y_nsamp,1);
% npid = 1:nx; npid(pid) = [];
Y_nsamp_pid = Y_nsamp(pid,:);
Y_nsamp_npid = Y_nsamp(npid,:);
Rpp = Y_nsamp_pid*Y_nsamp_pid';
Rpp(Rpp<0) = 0;
Rpp = log_modulus_transformation(Rpp);
Rnp = Y_nsamp_npid*Y_nsamp_pid';
Rnp(Rnp<0) = 0;
Rnp = log_modulus_transformation(Rnp);

Ysum = Y_nsamp*Y_nsamp_pid';
Ysum(Ysum<0) = 0;
Ysum = log_modulus_transformation(Ysum);
Rpp1 = sum(Ysum,1)';
Dpp = Rpp1;

Wpp = diag(Dpp)-Rpp;
Wnp = -Rnp;

D1 = Dpp;
Y1 = Y(pid,:);
Y2 = Y(npid,:);

if lambda
    %     W1 = Wpp;
    %     W2 = Wnp';
    %     Y1_new0 = -(W1\W2)*Y2;
    %     miny = min(Y1_new0(:));
    %     maxy = max(Y1_new0(:));
    %     dd = maxy-miny;
    
    W1 = Wpp+Kinv11*lambda*norm(Wpp)/norm(Kinv11);
    W2 = Wnp'+Kinv21'*lambda*norm(Wnp)/norm(Kinv21);
    Y1_new = -(W1\W2)*Y2;
    
    %     Y1_newa = Y1_new-min(Y1_new(:));
    %     Y1_newa = Y1_newa/max(Y1_newa(:));
    %     Y1_newa = Y1_newa*dd;
    %     Y1_new = Y1_newa+miny;
    
else
    W1 = Wpp;
    W2 = Wnp';
    Y1_new = -(W1\W2)*Y2;
end
f1 = sum(sum((Y1_new-fmu).*bsxfun(@times,D1,(Y1_new-fmu))));
f2 = sum(sum((Y1-fmu).*bsxfun(@times,D1,(Y1-fmu))));

if f1>f2 & rescale_flag
    disp('rescale')
    mm = f2/f1;
    Y1_new = (Y1_new-fmu)*mm+fmu;
end
Y1_new = vec(Y1_new);





