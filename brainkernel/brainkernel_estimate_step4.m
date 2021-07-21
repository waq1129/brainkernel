% step 4 for generating brain kernel: optimize MLE with initialization from
% step 3. It finalizes the PLS solution with MLE.

addpath(genpath(pwd)); warning off

%% load preprocessed data
load('brainkernel_step3.mat')

sp_flag = 0; % spetral representation of kernel, sp_flag=1 if nx is large; otherwise set sp_flag=0
npid_num = 500; % number of voxels selected to generate C22
lambda = 0; % We already impose the GP prior over the latent when we use PLS to estimate in step 3, 
            % here the performance of MAP is empirically better if we turn off the GP prior, 
            % which leads to a MLE estimator.
            
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
    [Kinv11, Kinv21] = return_Kinv_block(len_k,xgrid,pid,npid,large_flag,sp_flag,max_unit);
    if init_le_flag
        % pp0 = init_LaplaceEigen(fall_old+fmu,Y_nsamp,pid,rescale_flag,fmu(pid,:),lambda,xgrid,pgphyp_new);
        % [W,D] = gen_W_D(fall,Y,Ccov-diag(nsevar),log_flag,cov_flag);
        % pp0 = init_le1(fall_old+fmu,pid,nx,nf,W,D,KpriorInv,0,rescale_flag,fmu(pid,:));
        pp0 = init_LaplaceEigen_sp_Kinv(fall_old+fmu,Y_nsamp,pid,npid,rescale_flag,fmu(pid,:),lambda,Kinv11,Kinv21);
        
        pp0 = reshape(pp0,[],nf);
        pp0 = vec(pp0-fmu(pid,:));
        fnew = vec(fall_old(pid,:));
        pp1 = fnew(:);
        pp0 = pp0*lr+(1-lr)*pp1;
    else
        fnew = vec(fall_old(pid,:));
        pp0 = fnew(:);
    end
    
    Ynsamp_pid = Y_nsamp(pid,:);
    Ynsamp_npid = Y_nsamp(npid,:);
    Ccovblockpp = Ynsamp_pid*Ynsamp_pid';
    Ccovblockpp_noise = Ccovblockpp+nsevar_est*eye(length(pid));
    Ccovblockpn = Ynsamp_pid*Ynsamp_npid';
    
    cc_p = sum(Ccovblockpn.^2,1);
    [~,sort_id] = sort(cc_p,'descend');
    sort_id = sort_id(1:npid_num);
    npid1 = npid(sort_id);
    Ynsamp_npid1 = Y_nsamp(npid1,:);
    
    % MLE
    optid = [1,0,0,0];
    [ppinit,input_var] = input_var_pack(pp0,palpha_new,hyp,pgphyp_new,optid);
    
    k_rbf = kernelfun([-hyp(1)/2;hyp(2)/2], fall_old(npid1,:)+fmu(npid1,:));
    K_uu = k_rbf+nsevar_est*eye(length(npid1));
    [Ainv, U] = pdinv(K_uu);
    CinvY2 = Ainv*Ynsamp_npid1;
    
    lfunc = @(pp) map_negll(pp,input_var,pid,npid1,fall_old,nf,Ynsamp_pid,nsamp,CinvY2,nsevar_est,xgrid,kernelfun,lambda,Kinv11,Kinv21(sort_id,:));
    %lfunc = @(pp) mle_negll(pp,input_var,pid,fall_old,nf,nsevar_est,xgrid,kernelfun,Ccovblockpp_noise);
    %DerivCheck(lfunc,ppinit)
    [ppnew,fval] = minFunc(lfunc,ppinit,options);
    [~,~,fopt] = lfunc(ppnew);
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
            lfunc_gphyp = @(pp) hyp_opt_sp(pp,input_var,nf,xgrid,l_bound,Binvf1,Binvf2,Bf1,Bf2,wwnrm,dL1_B,large_flag);
            %DerivCheck(lfunc_gphyp,pgphyp0)
        else
            lfunc_gphyp = @(pp) hyp_opt(pp,input_var,nf,Ccov,nsamp,xgrid,kernelfun,lambda,KpriorInv);
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
    if mod(iter,100)==0
        save(['brainkernel_step4.mat'],'-v7.3')
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

save('brainkernel_latent.mat','rho_k', 'alpha', 'F', 'fmu', 'hyp', 'len_k', 'nc', 'xgrid','fall')

