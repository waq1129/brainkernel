function [W,D] = gen_W_D(Y0,Y,Ccov,log_flag,cov_flag)
if nargin<4
    log_flag = 0;
    cov_flag = 0;
end

if cov_flag
    R = Ccov;
    R(Ccov<0) = 0;
    R = log_modulus_transformation(R);
    R1 = sum(R,2);
    D = diag(R1);
    W = D-R;
    W = (W+W')/2;
    D = (D+D')/2;
else
    matrix_mse = @(x) sqrt(sum(vec(x.^2)));
    
    ee = [];
    llist = exp(-10:0.5:2);
    rho = mean(diag(Ccov));
    for ll=llist
        if log_flag
            R = kernel_dot_log(Y,Y,ll,rho,1);
        else
            R = kernel_dot(Y,Y,ll,rho);
        end
        ee = [ee;matrix_mse(R-Ccov)];
    end
    ii = find(ee==min(ee));
    ll = llist(ii);
    if log_flag
        R = kernel_dot_log(Y,Y,ll,rho,1);
    else
        R = kernel_dot(Y,Y,ll,rho);
    end
    R1 = sum(R,2);
    D = diag(R1);
    W = D-R;
    W = (W+W')/2;
    D = (D+D')/2;
    
end