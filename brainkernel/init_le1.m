function Y1a = init_le1(Y0,pid,nx,nf,W,D,Kinv,lambda,rescale_flag,fmu)
if nargin<7
    Kinv = 0;
    lambda = 0;
    rescale_flag = 0;
    fmu = 0;
end
pl = pid>nx; pid(pl) = mod(pid(pl),nx); pid(pid==0) = nx;

npid = 1:nx; npid(pid) = [];
np = length(pid);
rr = norm(W)/norm(Kinv)*20;
[W1, W2, W3] = blockwise(W+Kinv*lambda*rr,pid);
[D1, D2, D3] = blockwise(D,pid);

Y1 = Y0(pid,:);
Y2 = Y0(npid,:);

Y1a = -(W1\W2)*Y2;

f1 = trace((Y1a-fmu)'*D1*(Y1a-fmu));
f2 = trace((Y1-fmu)'*D1*(Y1-fmu));
% [f1 f2]

if f1>f2 & rescale_flag
    disp('rescale')
    mm = f2/f1;
    Y1a = (Y1a-fmu)*mm+fmu;
end
Y1a = vec(Y1a);


