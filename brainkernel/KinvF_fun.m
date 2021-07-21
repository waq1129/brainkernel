function [KinvF2,count] = KinvF_fun(b,covfunc,xgrid,subsizes,sigma2,gap,count)
display('count')
count

sub1 = subsizes(1);
sub2 = subsizes(2);
x1 = xgrid;
x2 = xgrid;
b1 = b(1:sub1, :);
b2 = b(sub1+1:end, :);

A = covfunc(x1(1:sub1,:),x2(1:sub1,:)); %M(1:sub1, 1:sub1);
A = A+sigma2*eye(sub1,sub1);
B = covfunc(x1(1:sub1,:),x2(sub1+1:end,:)); %M(1:sub1, sub1+1:end);
C = covfunc(x1(sub1+1:end,:),x2(1:sub1,:)); %M(sub1+1:end, 1:sub1);

if sub2<=gap
    display('enter here!!')
    D = covfunc(x1(sub1+1:end,:),x2(sub1+1:end,:)); %M(sub1+1:end, 1:sub1);
    D = D+sigma2*eye(size(D));
    
    DinvBT = D \ C;
    DinvF2 = D \ b2;
    
    count = count-1;
else
    subsizes1 = [gap, sub2-gap];
    [DinvBT,count] = KinvF_fun(C,covfunc,xgrid(sub1+1:end, :),subsizes1,sigma2,gap,count+1);
    [DinvF2,count] = KinvF_fun(b2,covfunc,xgrid(sub1+1:end, :),subsizes1,sigma2,gap,count+1);    
end

Kinv1 = [(A-B*DinvBT)\b1-(A\DinvBT')*((eye(sub2)-C*(A\DinvBT'))\b2)];
Kinv2 = [-DinvBT*((A-B*DinvBT)\b1)+DinvF2-DinvBT*((B*DinvBT-A)\(DinvBT'*b2))];
KinvF2 = [Kinv1; Kinv2];
