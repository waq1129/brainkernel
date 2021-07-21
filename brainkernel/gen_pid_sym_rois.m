function pidcell = gen_pid_sym_rois(xgrid, nc, nblock, max_unit)
left_id = find(xgrid(:,1)<0);
right_id = find(xgrid(:,1)>=0);

xgrid_left = xgrid(xgrid(:,1)<0,:);
xgrid_right = xgrid(xgrid(:,1)>=0,:);
% clf,hold on
% plot3(xgrid_right(:,1),xgrid_right(:,2),xgrid_right(:,3),'r.')
% plot3(xgrid_left(:,1),xgrid_left(:,2),xgrid_left(:,3),'k.')

pidcell_left = gen_pid(1,xgrid_left,size(xgrid_left,1),nc,nblock, max_unit);
pidcell_right = gen_pid(1,xgrid_right,size(xgrid_right,1),nc,nblock, max_unit);

pright = [];
for p=1:length(pidcell_right)
    pp = pidcell_right{p};
    pright = [pright; mean(xgrid_right(pp,:),1)];
end

pleft = [];
for p=1:length(pidcell_left)
    pp = pidcell_left{p};
    pleft = [pleft; mean(xgrid_left(pp,:),1)];
end

pright_id = 1:length(pright);
pleft_id = 1:length(pleft);
pleft_right = [];
for pp=pright_id
    xx = pright(pp,:);
    xx(1) = -xx(1);
    dd = repmat(xx,size(pleft,1),1)-pleft;
    dd = sum(dd.^2,2);
    [~,pleft_close] = min(dd);
    pleft_right = [pleft_right; pleft_close];
end

pidcell = cell(length(pleft_right),1);
for i=1:length(pidcell)
    pidcell_i = [right_id(pidcell_right{i}); left_id(pidcell_left{pleft_right(i)})];
    pidcell{i} = unique(pidcell_i);
end

pleft_right_non = setdiff(pleft_id', pleft_right);
pleft_right = [];
for pp=pleft_right_non'
    xx = pleft(pp,:);
    xx(1) = -xx(1);
    dd = repmat(xx,size(pright,1),1)-pright;
    dd = sum(dd.^2,2);
    [~,pleft_close] = min(dd);
    pleft_right = [pleft_right; pleft_close];
end
pidcell1 = cell(length(pleft_right),1);
for i=1:length(pidcell1)
    pidcell_i = [right_id(pidcell_right{pleft_right(i)}); left_id(pidcell_left{pleft_right_non(i)})];
    pidcell1{i} = unique(pidcell_i);
end

pidcell = [pidcell; pidcell1];


% for i=1:length(pidcell)
%     aa = cell2mat(pidcell(1:(i)));
%     aa = unique(aa);
%     clf,hold on
%     plot3(xgrid(:,1),xgrid(:,2),xgrid(:,3),'k.')
%     plot3(xgrid(aa,1),xgrid(aa,2),xgrid(aa,3),'ro')
%     %     view(-70,47)
%     drawnow,pause
% end
