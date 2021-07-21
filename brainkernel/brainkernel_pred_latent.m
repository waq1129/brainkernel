clc,clear,addpath(genpath(pwd)); warning off

%% implementation 1: get Kinv*(F-fmu)
load brainkernel_latent.mat
blocksizes = [30000,29412];
P = zeros(size(F,1),1);
Q = P';
Ps = {};
Qs = {};
covfunc = @(x1,x2) covSEiso_deriv(log([len_k;rho_k]), x1, x2, 0, 1);
sigma2 = 1e-3;

x1 = xgrid;
x2 = xgrid;
subsizes = blocksizes;
b = F-fmu;
sub1 = subsizes(1);

A = covfunc(x1(1:sub1,:),x2(1:sub1,:)); %M(1:sub1, 1:sub1);
A = A+sigma2*eye(sub1,sub1);
B = covfunc(x1(1:sub1,:),x2(sub1+1:end,:)); %M(1:sub1, sub1+1:end);
C = covfunc(x1(sub1+1:end,:),x2(1:sub1,:)); %M(sub1+1:end, 1:sub1);
D = covfunc(x1(sub1+1:end,:),x2(sub1+1:end,:)); %M(sub1+1:end, 1:sub1);
D = D+sigma2*eye(size(D));

b1 = b(1:sub1, :);
b2 = b(sub1+1:end, :);


Aib1 = A \ b1;
AiB = A \ B;

CAiB = C * AiB;
CAib1 = C * Aib1;
Zi = D - CAiB;

Zb2 = Zi\b2;
ZCAib1 = Zi\CAib1;
Wb1Xb2 = Aib1 + AiB * ZCAib1 - AiB * Zb2;
Yb1Zb2 = -ZCAib1 + Zb2;

KinvF = [Wb1Xb2; Yb1Zb2];

save('brainkernel_KinvF.mat','KinvF')

%% implementation 2: get Kinv*(F-fmu)
load brainkernel_latent.mat
gap = 5000;
blocksizes = [gap,size(F,1)-gap];
P = zeros(size(F,1),1);
Q = P';
Ps = {};
Qs = {};
covfunc = @(x1,x2) covSEiso_deriv(log([len_k;rho_k]), x1, x2, 0, 1);
sigma2 = 1e-3;

subsizes = blocksizes;
b = F-fmu;
count = 1;
KinvF = KinvF_fun(b,covfunc,xgrid,subsizes,sigma2,gap,count);

save('brainkernel_KinvF.mat','KinvF')

%% use KinvF to calculate the posterior mean for new voxels
xgrid_new = randn(10,3);
C_new = covfunc(xgrid_new, xgrid);
m_new = xgrid_new*reshape(alpha,nc,[]);
F_new = m_new+C_new'*KinvF;

save('brainkernel_latent_new_voxel.mat','F_new','xgrid_new')



