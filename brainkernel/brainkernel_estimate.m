function brainkernel_estimate()
%% load preprocessed data
load('brainkernel_pre.mat')

%% set up for optimization
% figure(4)
% set up flags
cov_flag = 1; % obsolete
log_flag = 0; % obsolete
rescale_flag = 0; % rescale in the initialization
init_le_flag = 1; % initialize the latent with laplace eigenmap
se_flag = 1; % optimize the brain kernel using PLS (se=1) or MAP (se=0)
change_flag = 0; % change the flags
lr_flag = 0; % anneal learning rate
alpha_flag = 0; % optimize the linear mapping alpha
len_flag = 0; % learn lengthscale
large_flag = 1; % operate with large matrix
sp_flag = 1; % spetral representation of kernel

% set up options
options = [];
options.Method='lbfgs';
options.TolFun=1e-3;
options.MaxIter = 1e2;
options.maxFunEvals = 1e2;
options.Display = 'off';
niter = 2e2;
matrix_mse = @(x) sqrt(sum(vec(x.^2)));
kernelfun = @covSEiso_deriv;
lambda = 1;
count = 1;
cchange = 2;
frac = 0.8;
lr = 1;

% define the block coordinate
nblock = 2;
pidcell = gen_pid(1,xgrid,nx,nc,nblock, max_unit);
if length(pidcell)==1, init_le_flag = 0; end

% intialize parameters
p0 = randn(nx*nf,1)*mean(abs(palpha(:)));
fall = reshape(p0,[],nf);
nsevar = nsevar_est*ones(nx,1);
fmu = xgrid*reshape(palpha,nc,[]);
palpha_new = palpha(:);
% len_k_est = exp(-bound_len(-2*log(len_k_est),minl,maxl)/2);
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

Y_nsamp = Y/sqrt(nsamp-1);

%% optimization
for iter = 1:niter
    iter
    ds(end)
    fall_old = fall;
    ll = mod(iter,length(pidcell)); if ll==0, ll=length(pidcell); end
    pid = pidcell{ll};
    np = length(pid);
    npid = 1:nx; npid(pid) = [];
    
    % initialization
    [Kinv11, Kinv21] = return_Kinv_block(len_k,xgrid,pid,large_flag,sp_flag,max_unit);
    if init_le_flag
        % pp0 = init_LaplaceEigen(fall_old+fmu,Y_nsamp,pid,rescale_flag,fmu(pid,:),lambda,xgrid,pgphyp_new);
        % [W,D] = gen_W_D(fall,Y,Ccov-diag(nsevar),log_flag,cov_flag);
        % pp0 = init_le1(fall_old+fmu,pid,nx,nf,W,D,KpriorInv,0,rescale_flag,fmu(pid,:));
        pp0 = init_LaplaceEigen_sp_Kinv(fall_old+fmu,Y_nsamp,pid,rescale_flag,fmu(pid,:),lambda,Kinv11,Kinv21);
        
        pp0 = reshape(pp0,[],nf);
        pp0 = vec(pp0-fmu(pid,:));
        fnew = vec(fall_old(pid,:));
        pp1 = fnew(:);
        if mod(count,cchange) == 0
            if lr_flag
                lr = lr*0.8;
            end
            count = 1;
            % change_flag = 1;
        end
        pp0 = pp0*lr+(1-lr)*pp1;
    else
        fnew = vec(fall_old(pid,:));
        pp0 = fnew(:);
        if mod(count,cchange) == 0
            count = 1;
            % change_flag = 1;
        end
    end
    Ccovblockpp = Y_nsamp(pid,:)*Y_nsamp(pid,:)';
    Ccovblockpn = Y_nsamp(pid,:)*Y_nsamp(npid,:)';
    
    if se_flag
        % PLS
        S1 = Ccovblockpp-nsevar_est*eye(length(pid));
        S2 = Ccovblockpn';
        optid = [1,alpha_flag,0,0];
        [ppinit,input_var] = input_var_pack(pp0,palpha_new,hyp,pgphyp_new,optid);
        lfunc = @(pp) compNegEmbeddedGPlogLi_gplvm_bcg_se_large(pp,input_var,pid,fall_old,nf,xgrid,kernelfun,lambda,Kinv11,Kinv21,S1,S2,nsamp);
        %DerivCheck(lfunc,ppinit)
        [ppnew,fval] = minFunc(lfunc,ppinit,options);
        [~,~,fopt] = lfunc(ppnew);
        if alpha_flag
            [~,palpha_new,~,~,~] = input_var_unpack(ppnew,input_var,nc,nf,np);
            fmu = xgrid*reshape(palpha_new,nc,[]);
            palphas = [palphas; palpha_new(:)'];
        end
    else
        % MAP
        k_rbf = kernelfun([-hyp(1)/2;hyp(2)/2], fall_old+fmu);
        K_uu = k_rbf+diag(nsevar);
        [Ainv, U] = pdinv(K_uu);
        optid = [1,0,0,0];
        [ppinit,input_var] = input_var_pack(pp0,palpha_new,hyp,pgphyp_new,optid);
        lfunc = @(pp) compNegEmbeddedGPlogLi_gplvm_bcg_delta_large(pp,input_var,pid,fall_old,nf,Ccov,nsamp,Ainv,xgrid,kernelfun,lambda,Kinv11,Kinv21);
        %DerivCheck(lfunc,ppinit)
        [ppnew,fval] = minFunc(lfunc,ppinit,options);
        [~,~,fopt] = lfunc(ppnew);
    end
    fopt = reshape(fopt,[],nf);
    fall(pid,:) = fopt;
    
    %%%%%%%%%%%%%%%%
    % learn len_k
    if len_flag
        pf1 = fall;
        palpha1 = palpha_new;
        phyp1 = hyp;
        pgphyp1 = pgphyp_new;
        optid = [0,0,0,1];
        [pgphyp0,input_var] = input_var_pack(pf1,palpha1,phyp1,pgphyp1,optid);
        
        % Set bounds for optimizing hypers
        l_bound = [-log(exp(-pgphyp_new/2)/frac)*2, -log(exp(-pgphyp_new/2)*frac)*2]; % bounds for rho
        l_bound(1) = max([-log(maxl)*2, l_bound(1)]);
        l_bound(2) = min([-log(minl)*2, l_bound(2)]);
        
        if sp_flag
            % collect statistics for opt len
            f1 = fall(pid,:);
            f2 = fall(npid,:);
            xposcell = mat2tiles(xgrid,[nx,1]);
            [Bmats,Cdiag,cdiagvecs,wwnrm] = myfft_nu(xposcell,exp(-pgphyp0/2),1,exp(-l_bound(2)/2),exp(-l_bound(1)/2),large_flag);
            BB = column_kron(Bmats);
            BB11 = BB(:,pid);
            BB22 = BB(:,npid);
            Bf1 = BB11*f1;
            Bf2 = BB22*f2;
            Binvf1 = (BB11*BB11'+nn*eye(size(BB11,1)))\Bf1;
            Binvf2 = (BB22*BB22'+nn*eye(size(BB22,1)))\Bf2;
            BB11_2 = BB11*BB11';
            BB22_2 = BB22*BB22';
            
            dL1_B = sum(Bf1.*Bf1,2)+2*sum(Bf1.*Bf2,2)+sum(Bf2.*Bf2,2);
            % lfunc_gphyp = @(pp) compNegEmbeddedGPlogLi_gplvm_se_len_sp_B3(pp,input_var,nf,xgrid,l_bound,BB11_2,BB22_2,Bf1,Bf2,wwnrm,large_flag);
            lfunc_gphyp = @(pp) compNegEmbeddedGPlogLi_gplvm_se_len_sp_B1(pp,input_var,nf,xgrid,l_bound,Binvf1,Binvf2,Bf1,Bf2,wwnrm,dL1_B,large_flag);
            %DerivCheck(lfunc_gphyp,pgphyp0)
        else
            lfunc_gphyp = @(pp) compNegEmbeddedGPlogLi_gplvm_se_len(pp,input_var,nf,Ccov,nsamp,xgrid,@covSEiso_deriv,lambda,KpriorInv);
            %DerivCheck(lfunc_gphyp,pgphyp0)
        end
        
        options_hypers = optimset('display', 'off', 'maxIter', 1e3, ...
            'Algorithm', 'interior-point', 'TolFun', 1e-5, 'TolX', 1e-3);
        pgphyp_new = fmincon(lfunc_gphyp,pgphyp0,[],[],[],[],l_bound(1),l_bound(2),[],options_hypers);
        pgphyps = [pgphyps; pgphyp_new];
        
        len_k = pgphyp_new; % marginal variance
        rho_k = 0;
    end
    %%%%%%%%%%%%%%%%
    
    C_bk = kernelfun([-hyp(1)/2;hyp(2)/2], fall(pid,:)+fmu(pid,:));
    %subplot(221),imagesc(C_bk),axis image,colorbar,drawnow,title(['Cmat:' num2str(matrix_mse(Ccovblockpp-C_bk)) ' Ccov:' num2str(matrix_mse(Ccovblockpp-C_bk)) ' iter: ' num2str(iter) ' gap:' num2str(pid(1)) '-' num2str(pid(end))]);
    %subplot(222),imagesc(Ccovblockpp),axis image,colorbar,drawnow,title('true covariance')
    %subplot(223); plot([normcol(fall+fmu)]); title('f brain kernel'); axis tight; axis square;drawnow
    
    C_alpha = kernelfun([-hyp(1)/2;hyp(2)/2], fmu(pid,:));
    %subplot(224),imagesc(C_alpha),axis image,colorbar,drawnow,title('current Kalpha, linear projection')
    
    ds = [ds; matrix_mse(Ccovblockpp-C_bk)+2*matrix_mse(Ccovblockpn ...
        -kernelfun([-hyp(1)/2;hyp(2)/2], fall(pid,:)+fmu(pid,:),fall(npid,:)+fmu(npid,:),0,1))];
    if min(ds)==ds(end)
        fall_min = fall;
        palpha_min = palpha_new;
        pgphyp_min = pgphyp_new;
    end
    if ll==length(pidcell)
        count = count+1;
    end
    if change_flag
        disp('change_flag is 1')
        %         se_flag = 0;
        %         init_le_flag = 0;
        %         rescale_flag = 1;
        alpha_flag = 0;
        len_flag = 0;
    end
    if mod(iter,10)==0
        %    save([nname '_' num2str(max_unit1) '_midresult' num2str(iter) '.mat'],'-v7.3')
    end
    if 0%abs(ds(end)-ds(end-1))<1e-2 %& change_flag
        break;
    end
end
fall = fall_min;
palpha_new = palpha_min;
pgphyp_new = pgphyp_min;
fmu = xgrid*reshape(palpha_new,nc,[]);

hyps.len_C = exp(-hyp(1)/2);
hyps.rho_C = exp(hyp(2)/2);
hyps.nsevar = nsevar_est;
hyps.len_K = exp(-pgphyp_new/2);
hyps.rho_K = rho_k_est;
hyps.alpha = reshape(palpha_new,nc,[]);

%% collect and save to mat
hyp = hyps;
F = fall+fmu; % the latent to construct the brain kernel covariance function
alpha = hyps.alpha; % the linear mapping for fmu
rho_k = hyps.rho_K;
len_k = hyps.len_K;

save('brainkernel_prior.mat','rho_k', 'alpha', 'F', 'fmu', 'hyp', 'len_k', 'nc', 'xgrid')

