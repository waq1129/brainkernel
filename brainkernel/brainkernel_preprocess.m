% preprocess for generating brain kernel
function brainkernel_preprocess()

% clc,clear,addpath(genpath(pwd)); warning off
data = load('../HCP_data/wholebrain_rfMRI.mat'); % load data (leading eigen vectors of the covariance matrix) from HCP resting state fMRI
wholebrain = double(data.wholebrain);
xgrid = data.coords;

% number of latent dimensions
nf = 20;

% zscore fMRI data: 59412 voxels, 4800 eigenvectors
Y = zscore(wholebrain')';
Ccov = cov(Y');
Cmat = Ccov;
Cmat = Cmat/Cmat(1);
fsamp = Y(:,1:nf);

matrix_mse = @(x) sqrt(sum(vec(x.^2)));
[nx,nsamp] = size(Y);
max_unit = 2000; % block/unit size
min_nblock = ceil(nx/max_unit); % number of blocks
nc = size(xgrid,2);
minl = 10;
xwid = range(xgrid);
maxl = min(xwid);
nn = 1e-6;

%% estimate for hyp_c (hyper parameters for covariance)
Y_nsamp = Y/sqrt(nsamp-1);
diagS = sum(Y_nsamp.^2,2);
pidcell = mat2tiles([1:nx]',[min([nx,max_unit]),1]);
maxCcov = -inf;
for ii=1:length(pidcell)
    pid1 = pidcell{ii};
    for jj=ii:length(pidcell)
        pid2 = pidcell{jj};
        Ccov_ij = Y_nsamp(pid1,:)*Y_nsamp(pid2,:)';
        if ii==jj
            Ccov_ij(logical(eye(size(Ccov_ij)))) = [];
        end
        mc = max(vec(Ccov_ij));
        if maxCcov<mc
            maxCcov = mc;
        end
    end
end
mdiagS = mean(diagS);
nsevar_est = mdiagS-maxCcov;
if nsevar_est<0
    [~, ee] = svd(Y_nsamp,'econ');
    de = diag(ee).^2;
    cde = cumsum(de);
    cc = cde<cde(end)*0.95;
    id_cc = find(cc==1);
    nfln = id_cc(end)+1;
    ll = de(nfln:end);
    nsevar_est = sum(ll)/(length(ll)+max([0,nx-nsamp]));
end
rho_est = mdiagS-nsevar_est;
len_est = 1;%min(range(xgrid))/2;
hyp = [-2*log(len_est);log(rho_est);log(nsevar_est)];

%% estimate for hyp_k (hyper parameters for kernel)
% estimate len_k
step = 100;
pidcell = mat2tiles([1:nx]',[step,1]);
sigmas = [];
for iter = 1:length(pidcell)
    ll = mod(iter,length(pidcell)); if ll==0, ll=length(pidcell); end
    pid = pidcell{ll};
    if length(pid)>1
        Ccovblock = Y_nsamp(pid,:)*Y_nsamp(pid,:)'-nsevar_est*eye(length(pid));
        sigma = empirical_len(Ccovblock,xgrid(pid,:),0);
        sigmas = [sigmas; sigma];
    end
end
sigma = median(sigmas);
sigma
len_k_est = sigma;
len_k_est = exp(-bound_len(-2*log(len_k_est),minl,maxl)/2);
rho_k_est = 1; % marginal variance rho_k

% %visualize the GP kernel with hyp_k
%subplot(222),
%vis_block(@(pid1,pid2) covSEiso([log(len_k_est),log(rho_k_est)],xgrid(pid1,:),xgrid(pid2,:)),nx,min_nblock)
% title('Kfprior\_ij')

%% find the best fit se kernel
% step = round(nx/min_nblock);
% pidcell = mat2tiles([1:nx]',[step,1]);
sigma = len_k_est;
nblock = 2;
pidcell = gen_pid(sigma,xgrid,nx,nc,nblock);
llist = linspace(-5,5,10);
eeall = [];
for ll=llist
    ll
    ee = [];
    for ii = 1:length(pidcell)
        pid1 = pidcell{ii};
        for jj=ii%:length(pidcell)
            pid2 = pidcell{jj};
            Kfprior_ij = covSEiso_deriv([ll;0],xgrid(pid1,:),xgrid(pid2,:),0,1); % the covariance
            if ii==jj
                Ccovblock = Y_nsamp(pid1,:)*Y_nsamp(pid2,:)'-nsevar_est*eye(length(pid1));
                Ccovblock(Ccovblock<0) = 0;
                ee = [ee; matrix_mse(Ccovblock-Kfprior_ij)];
            else
                Ccovblock = Y_nsamp(pid1,:)*Y_nsamp(pid2,:)';
                Ccovblock(Ccovblock<0) = 0;
                ee = [ee; 2*matrix_mse(Ccovblock-Kfprior_ij)];
            end
        end
    end
    eeall = [eeall; mean(ee)];
end
[~,ii] = min(eeall);
len_se = exp(llist(ii));

% %visualize the best se kernel
%subplot(223),
%vis_block(@(pid1,pid2) covSEiso_deriv([log(len_se);0],xgrid(pid1,:),xgrid(pid2,:),0,1),nx,min_nblock)
% title('Kse\_ij')

%% estimate for alpha
sigma = 1;
nblock = 2;
pidcell = gen_pid(sigma,xgrid,nx,nc,nblock);
alpha_init = 0.001*randn(nc,nf)/len_se;
alphas = [];
hyp(1) = 0;
for iter = 1:2*length(pidcell)
    iter
    ll = mod(iter,length(pidcell)); if ll==0, ll=length(pidcell); end
    pid = pidcell{ll};
    Ccovblock = Y_nsamp(pid,:)*Y_nsamp(pid,:)';
    xgrid_block = xgrid(pid,:);
    %     f_alpha = @(alpha) trace((Ccovblock - kSE(1,1,xgrid_block*reshape(alpha,nc,[]))-nsevar_est*eye(length(pid)))'*(Ccovblock - kSE(1,1,xgrid_block*reshape(alpha,nc,[]))-nsevar_est*eye(length(pid))));
    %     options = optimset('GradObj','off','display', 'off', 'largescale', 'off','maxiter',1e6,'maxfunevals',1e6);
    %     alpha_new = fminunc(f_alpha,alpha_init(:),options);
    f_alpha = @(alpha) learn_alpha(alpha,xgrid_block,Ccovblock,hyp(1:2),nsevar_est);
    options = optimset('GradObj','on','display', 'off', 'largescale', 'off','maxiter',1e6,'maxfunevals',1e6);
    alpha_new = fminunc(f_alpha,alpha_init(:),options);
    %subplot(224),imagesc([Ccovblock,covSEiso_deriv([-hyp(1)/2;hyp(2)/2],xgrid_block*reshape(alpha_new,nc,[]))]),colorbar,axis image,title(iter),drawnow
    alpha_init = alpha_new;
    alphas = [alphas alpha_new(:)];
end
palpha = mean(alphas,2);

% %visualize the best alpha kernel
%subplot(224),
%vis_block(@(pid1,pid2) covSEiso([-hyp(1)/2;hyp(2)/2],xgrid(pid1,:)*reshape(palpha,nc,[]),xgrid(pid2,:)*reshape(palpha,nc,[])),nx,min_nblock)
% title('Kalpha\_ij')

%% set other parameters
lambda = 1;
iters = 100;
display = 1;
mflag = 1;
if mflag
    options = [];
    options.Method='lbfgs';
    options.TolFun=1e-10;
    options.MaxIter = 2e3;
    options.maxFunEvals = 2e3;
    options.Display = 'off';
else
    options = optimset('GradObj','off','display', 'off', 'largescale', 'off','maxiter',1e6,'maxfunevals',1e6);
end

save('brainkernel_pre','-v7.3')
