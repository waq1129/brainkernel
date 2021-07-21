% step 2 for generating brain kernel: optimize simple mean squared loss
% with initialization from Y_nsamp

addpath(genpath(pwd)); warning off

%% load step 1 data
load('brainkernel_step1.mat')
Y_nsamp = Y/sqrt(nsamp-1);

%% set up for optimization
% set up flags
cov_flag = 1; % obsolete
log_flag = 0; % obsolete
rescale_flag = 0; % rescale in the initialization
init_le_flag = 0; % initialize the latent with laplace eigenmap
se_flag = 1; % optimize the brain kernel using PLS (se=1) or MAP (se=0)
change_flag = 0; % change the flags
lr_flag = 0; % anneal learning rate
alpha_flag = 0; % optimize the linear mapping alpha
len_flag = 0; % learn lengthscale
large_flag = 1; % operate with large matrix
sp_flag = 0; % spetral representation of kernel

% set up options
options = [];
options.Method = 'lbfgs';
options.TolFun = 1e-3;
options.MaxIter = 1e2;
options.maxFunEvals = 1e2;
options.Display = 'on';
matrix_mse = @(x) sqrt(sum(vec(x.^2)));
kernelfun = @covSEiso_deriv;%@covMatern3iso_deriv;%
lambda = 1;
count = 1;
cchange = 2;
frac = 0.8;
lr = 1;

% define the block coordinate
nblock = 3;
max_unit = 5000;
pidcell = gen_pid_sym_rois(xgrid, nc, nblock, max_unit);
if length(pidcell)==1, init_le_flag = 0; end

% intialize parameters
p0 = randn(nx*nf,1)*mean(abs(palpha(:)));
nsevar_est = 1e-3;
nsevar = nsevar_est*ones(nx,1);
fmu = xgrid*reshape(palpha,nc,[]);
fall = Y_nsamp(:,1:nf)-fmu;
palpha_new = palpha(:);
pgphyp_new = -2*log(len_k_est);
palphas = palpha_new(:)';
pgphyps = pgphyp_new;
ds = inf;
fall_min = 0;
palpha_min = 0;
pgphyp_min = 0;

% initial KpriorInv
len_k = pgphyp_new;
rho_k = 0;

niter = 2*length(pidcell);

%% optimization
for iter = 1:niter
    iter
    ds(end)
    fall_old = fall;
    fmu = xgrid*reshape(palpha_new,nc,[]);
    ll = mod(iter,length(pidcell)); if ll==0, ll=length(pidcell); end
    pid = pidcell{ll};
    np = length(pid);
    npid = 1:nx; npid(pid) = [];
    
    % initialization
    fnew = vec(fall_old(pid,:));
    pp0 = fnew(:);
    
    Ynsamp_pid = Y_nsamp(pid,:);
    Ynsamp_npid = Y_nsamp(npid,:);
    Ccovblockpp = Ynsamp_pid*Ynsamp_pid';
    Ccovblockpn = Ynsamp_pid*Ynsamp_npid';
    
    % MSE
    optid = [1,alpha_flag,0,0];
    [ppinit,input_var] = input_var_pack(pp0,palpha_new,hyp,pgphyp_new,optid);
    lfunc = @(pp) mse_diagonalonly(pp,input_var,pid,fall_old,nf,xgrid,kernelfun,Ccovblockpp);
    %DerivCheck(lfunc,ppinit)
    [ppnew,fval] = minFunc(lfunc,ppinit,options);
    [~,~,fopt] = lfunc(ppnew);
    fopt = reshape(fopt,[],nf);
    fall(pid,:) = fopt;
    
    C_bk = kernelfun([-hyp(1)/2;hyp(2)/2], fall(pid,:)+fmu(pid,:));
    C_bkpn = kernelfun([-hyp(1)/2;hyp(2)/2], fall(npid,:)+fmu(npid,:), fall(pid,:)+fmu(pid,:));
    
    subplot(321),imagesc(C_bk),axis image,colorbar,title(['Cmat-Ccov:' num2str(matrix_mse(Ccovblockpp-C_bk)) ' iter: ' num2str(iter) ' gap:' num2str(pid(1)) '-' num2str(pid(end))]);
    subplot(322),imagesc(Ccovblockpp),axis image,colorbar,title('true covariance')
    subplot(323),imagesc(C_bkpn, 'XData', [1 500], 'YData', [1 500]),axis image,colorbar,title(['Cmat-Ccov:' num2str(matrix_mse(Ccovblockpn'-C_bkpn))]);
    subplot(324),imagesc(Ccovblockpn', 'XData', [1 500], 'YData', [1 500]),axis image,colorbar,title('true cross covariance')
    subplot(325); plot([normcol(fall(pid,:)+fmu(pid,:))]); title('f brain kernel'); axis tight; axis square;
    C_alpha = kernelfun([-hyp(1)/2;hyp(2)/2], fmu(pid,:));
    subplot(326),imagesc(C_alpha),axis image,colorbar,title('current C\_alpha, linear projection'),drawnow
    
    ds = [ds; matrix_mse(Ccovblockpp-C_bk)+2*matrix_mse(Ccovblockpn'-C_bkpn)];
    if min(ds)==ds(end)
        fall_min = fall;
        palpha_min = palpha_new;
        pgphyp_min = pgphyp_new;
    end
    if mod(iter,50)==0
        save(['brainkernel_step2.mat'],'-v7.3')
    end
end
save(['brainkernel_step2.mat'],'-v7.3')
