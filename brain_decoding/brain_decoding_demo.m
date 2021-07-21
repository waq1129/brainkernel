% brain decoding demo: run a binary classification task on the working
% memory task fMRI from HCP. The demo runs for one binary pair for subject 100307 and returns
% the accuracy performance comparing among ridge, se (squared exponential) 
% and bk (brain kernel).

clc,clear
addpath(genpath(pwd)); warning off
rng('shuffle')

% binary classification: class 1 vs 2
l1 = '1'; % class index
l2 = '2'; % class index
lrflag = '0'; % which session to train and test: lrflag=1, left-to-right session for train, right-to-left session for test; lrflag=0, the other way around
tvflag = '1';

subname = ['../HCP_data/hcp_sub100307_WM.mat'];
subname
tvflag = str2num(tvflag)
lrflag = str2num(lrflag)
l1 = str2num(l1)
l2 = str2num(l2)

% load working memory task fMRI from HCP
load('../HCP_data/wm_mask.mat')
load(subname)
X = wholebrain(logical(mask),:);
xgrid = coords(logical(mask),:);
d = size(X,1);


%%
gg = uint32(size(X,2)/2);
if lrflag
    [Xtr,muX,sigmaX] = zscore(X(:,1:gg)'); Xtr = Xtr';
    Xte = X(:,gg+1:gg*2);
    Xte = Xte-repmat(muX',1,size(Xte,2));
    Xte = Xte./repmat(sigmaX',1,size(Xte,2));
    ytr = labels(1:gg);
    yte = labels(gg+1:gg*2);
else
    [Xtr,muX,sigmaX] = zscore(X(:,gg+1:gg*2)'); Xtr = Xtr';
    Xte = X(:,1:gg);
    Xte = Xte-repmat(muX',1,size(Xte,2));
    Xte = Xte./repmat(sigmaX',1,size(Xte,2));
    ytr = labels(gg+1:gg*2);
    yte = labels(1:gg);
end

if l1==l2
    display('same class')
    return
end
ii1tr = ytr==l1;
ii2tr = ytr==l2;
ii1te = yte==l1;
ii2te = yte==l2;

x_train = [Xtr(:, ii1tr) Xtr(:, ii2tr)]';
y_train = [-1*ones(sum(ii1tr),1); ones(sum(ii2tr),1)];
x_test = [Xte(:, ii1te) Xte(:, ii2te)]';
y_test = [-1*ones(sum(ii1te),1); ones(sum(ii2te),1)];
% [x_train,y_train] = perturb_data(x_train,y_train);
% [x_test,y_test] = perturb_data(x_test,y_test);
x_train = zscore(x_train);
x_test = zscore(x_test);
% [x_train, xmu, xsigma] = zscore(x_train);
% x_test = (x_test-repmat(xmu,size(x_test,1),1))./repmat(xsigma,size(x_test,1),1);

k = 3;
c = cvpartition(size(x_train,1),'kfold',k);

nll = 100;
groupsize = 100;

clear x1 x2 y1 y2 x y X X1 X2 wholebrain x1_train x2_train y1_train y2_train x1_test x2_test y1_test y2_test x1_valid x2_valid y1_valid y2_valid Xtr Xte ytr yte

%% ridge
nlist = linspace(-10,10,nll);
acc_valid_all1 = zeros(length(nlist),k);
for kk=1:k
    for ii = 1:length(nlist)
        display(['ridge, fold=' num2str(kk) ', ii=' num2str(ii)])
        if tvflag
            x_train0 = x_train(c.training(kk),:);
            y_train0 = y_train(c.training(kk));
            x_valid0 = x_train(c.test(kk),:);
            y_valid0 = y_train(c.test(kk));
        else
            x_valid0 = x_train(c.training(kk),:);
            y_valid0 = y_train(c.training(kk));
            x_train0 = x_train(c.test(kk),:);
            y_train0 = y_train(c.test(kk));
        end
        x_train0 = zscore(x_train0);
        x_valid0 = zscore(x_valid0);
        
        xx = x_train0*x_train0';
        
        % w = (x_train'*x_train+eye(d)*exp(-nlist(ii)))\(x_train'*y_train);
        w = x_train0'*((xx+eye(size(x_train0,1))*exp(-nlist(ii)))\y_train0);
        y_train_pred = sign(x_train0*w);
        y_valid_pred = sign(x_valid0*w);
        
        acc_valid = sum(y_valid_pred==y_valid0)/length(y_valid_pred);
        acc_valid_all1(ii,kk) = acc_valid;
    end
end
acc_valid_all1 = mean(acc_valid_all1,2);
[aa1,bb1] = find(acc_valid_all1==max(vec(acc_valid_all1)));
w = gausswin(nll-1,20);
acc_valid_all11 = filter_win(acc_valid_all1,w);
[aa11,bb11] = find(acc_valid_all11==max(vec(acc_valid_all11)));
acc_valid_all12 = conv(acc_valid_all1,w,'same');
[aa12,bb12] = find(acc_valid_all12==max(vec(acc_valid_all12)));
aa1 = [aa1; aa11; aa12];
bb1 = [bb1; bb11; bb12];
hyp1 = exp(-nlist(aa1))

xx = x_train*x_train';
acc_test10 = [];
for dd=1:length(aa1)
    w1 = x_train'*((xx+eye(size(x_train,1))*exp(-nlist(aa1(dd))))\y_train);
    y_test_pred1 = sign(x_test*w1);
    acc_test10 = [acc_test10; sum(y_test_pred1==y_test)/length(y_test_pred1)];
end
acc_test1 = max(acc_test10)

%% se
rlist = exp(linspace(-10,10,nll));
llist = exp(linspace(-5,5,nll));
acc_valid_all2 = zeros(length(llist),length(rlist), k);
acc_test_all2 = zeros(length(llist),length(rlist));

for kk=1:k
    if tvflag
        x_train0 = x_train(c.training(kk),:);
        y_train0 = y_train(c.training(kk));
        x_valid0 = x_train(c.test(kk),:);
        y_valid0 = y_train(c.test(kk));
    else
        x_valid0 = x_train(c.training(kk),:);
        y_valid0 = y_train(c.training(kk));
        x_train0 = x_train(c.test(kk),:);
        y_train0 = y_train(c.test(kk));
    end
    x_train0 = zscore(x_train0);
    x_valid0 = zscore(x_valid0);
    
    for ii = 1:length(llist)
        display(['se, fold=' num2str(kk) ', ii=' num2str(ii)])
        covfun = @(x,y) covSEiso_deriv(log([llist(ii);1]),x,y,0,1);
        if kk==1
            [Klx,Klx1] = cov_block_mult_x(xgrid, x_train0', covfun, groupsize, x_train');
        else
            Klx = cov_block_mult_x(xgrid, x_train0', covfun, groupsize);
        end
        
        for jj = 1:length(rlist)
            % w = (x_train'*x_train+Kinv)\(x_train'*y_train);
            Kx = Klx;
            xKx = x_train0*Kx;
            w = Kx*((eye(size(x_train0,1))/rlist(jj)^2+xKx)\(y_train0));
            y_train_pred = sign(x_train0*w);
            y_valid_pred = sign(x_valid0*w);
            
            acc_valid = sum(y_valid_pred==y_valid0)/length(y_valid_pred);
            acc_valid_all2(ii,jj,kk) = acc_valid;
            
            if kk==1
                Kx = Klx1;
                xKx = x_train*Kx;
                w = Kx*((eye(size(x_train,1))/rlist(jj)^2+xKx)\(y_train));
                y_test_pred = sign(x_test*w);
                acc_test = sum(y_test==y_test_pred)/length(y_test_pred);
                acc_test_all2(ii,jj) = acc_test;
            end
        end
    end
end
acc_valid_all2 = mean(acc_valid_all2,3);
[aa2,bb2] = find(acc_valid_all2==max(vec(acc_valid_all2)));
w = gausswin(nll-1,20)*gausswin(nll-1,20)';
acc_valid_all21 = filter_win(acc_valid_all2,w);
[aa21,bb21] = find(acc_valid_all21==max(vec(acc_valid_all21)));
acc_valid_all22 = conv2(acc_valid_all2,w,'same');
[aa22,bb22] = find(acc_valid_all22==max(vec(acc_valid_all22)));
aa2 = [aa2; aa21; aa22];
bb2 = [bb2; bb21; bb22];
ab = unique([aa2 bb2],'rows');
aa2 = ab(:,1);
bb2 = ab(:,2);
hyp2 = [llist(aa2);rlist(bb2)]

acc_test20 = [];
for dd=1:length(aa2)
    acc_test20 = [acc_test20; acc_test_all2(aa2(dd),bb2(dd))];
end
acc_test2 = max(acc_test20)


%% bk
load ../brainkernel/brainkernel_latent.mat
F_bk = F(logical(mask),:);
rlist = exp(linspace(-10,10,nll));
llist = exp(linspace(-5,5,nll));
acc_valid_all3 = zeros(length(llist),length(rlist), k);
acc_test_all3 = zeros(length(llist),length(rlist));

for kk=1:k
    if tvflag
        x_train0 = x_train(c.training(kk),:);
        y_train0 = y_train(c.training(kk));
        x_valid0 = x_train(c.test(kk),:);
        y_valid0 = y_train(c.test(kk));
    else
        x_valid0 = x_train(c.training(kk),:);
        y_valid0 = y_train(c.training(kk));
        x_train0 = x_train(c.test(kk),:);
        y_train0 = y_train(c.test(kk));
    end
    x_train0 = zscore(x_train0);
    x_valid0 = zscore(x_valid0);
    
    for ii = 1:length(llist)
        display(['se, fold=' num2str(kk) ', ii=' num2str(ii)])
        covfun = @(x,y) covSEiso_deriv(log([llist(ii);1]),x,y,0,1);
        if kk==1
            [Klx,Klx1] = cov_block_mult_x(F_bk, x_train0', covfun, groupsize, x_train');
        else
            Klx = cov_block_mult_x(F_bk, x_train0', covfun, groupsize);
        end
        
        for jj = 1:length(rlist)
            % w = (x_train'*x_train+Kinv)\(x_train'*y_train);
            Kx = Klx;
            xKx = x_train0*Kx;
            w = Kx*((eye(size(x_train0,1))/rlist(jj)^2+xKx)\(y_train0));
            y_train_pred = sign(x_train0*w);
            y_valid_pred = sign(x_valid0*w);
            
            acc_valid = sum(y_valid_pred==y_valid0)/length(y_valid_pred);
            acc_valid_all3(ii,jj,kk) = acc_valid;
            
            if kk==1
                Kx = Klx1;
                xKx = x_train*Kx;
                w = Kx*((eye(size(x_train,1))/rlist(jj)^2+xKx)\(y_train));
                y_test_pred = sign(x_test*w);
                acc_test = sum(y_test==y_test_pred)/length(y_test_pred);
                acc_test_all3(ii,jj) = acc_test;
            end
        end
    end
end
acc_valid_all3 = mean(acc_valid_all3,3);
[aa3,bb3] = find(acc_valid_all3==max(vec(acc_valid_all3)));
w = gausswin(nll-1,20)*gausswin(nll-1,20)';
acc_valid_all31 = filter_win(acc_valid_all3,w);
[aa31,bb31] = find(acc_valid_all31==max(vec(acc_valid_all31)));
acc_valid_all33 = conv2(acc_valid_all3,w,'same');
[aa33,bb33] = find(acc_valid_all33==max(vec(acc_valid_all33)));
aa3 = [aa3; aa31; aa33];
bb3 = [bb3; bb31; bb33];
ab = unique([aa3 bb3],'rows');
aa3 = ab(:,1);
bb3 = ab(:,2);
hyp3 = [llist(aa3);rlist(bb3)]

acc_test30 = [];
for dd=1:length(aa3)
    acc_test30 = [acc_test30; acc_test_all3(aa3(dd),bb3(dd))];
end
acc_test3 = max(acc_test30)

%%
acc_test = [acc_test1 acc_test2 acc_test3]




