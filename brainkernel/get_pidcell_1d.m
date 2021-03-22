function pidcell1 = get_pidcell_1d(len_true_k, nx1, nblock)
ni1 = round(nx1/nblock);
if ni1<=len_true_k*2
    ni1 = len_true_k*2+1;
end
step = max([min([ni1-len_true_k*2,ni1]),1]);
ii = [1:step:nx1]';
jj = [0:ni1-1];
kk = repmat(ii,1,ni1)+repmat(jj,size(ii,1),1);
kii = kk(:,end)<=nx1;
kii(sum(kii)+1) = 1;
kk = kk(kii,:);
gs = size(kk,2);
kk = kk';
kk(kk>nx1) = [];
pidcell1 = mat2tiles(kk,[1,gs]);