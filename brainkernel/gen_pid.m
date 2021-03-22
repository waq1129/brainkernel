function pidcell = gen_pid(len_true_k,xgrid,nx_true,nc,nblock,max_unit)
if nargin<6
    max_unit = 2000;
end
switch nc
    case 1
        %         nx = length(xgrid);
        %         ni = round(nx/nblock);
        %         gap = ni;
        %         len_true_k = round(max([len_true_k 1]));
        %         step = max([min([gap-len_true_k*2,gap]),1]);
        %         ii = [1:step:nx]';
        %         jj = [0:gap-1];
        %         kk = repmat(ii,1,gap)+repmat(jj,size(ii,1),1);
        %         kii = kk(:,end)<=nx;
        %         kii(sum(kii)+1) = 1;
        %         kk = kk(kii,:);
        %         gs = size(kk,2);
        %         kk = kk';
        %         kk(kk>nx) = [];
        %         pidcell = mat2tiles(kk,[1,gs]);
        nx = ceil(range(xgrid));
        xx = linspace(min(xgrid),max(xgrid),nx);
        len_true_k = round(max([len_true_k 1]));
        pidcell1 = get_pidcell_1d(len_true_k, nx, nblock);
        
        pidcell = cell(length(pidcell1),1);
        mg = round(size(xgrid,1)/nblock);
        cc = 1;
        ll = 1:size(xgrid,1);
        vv = zeros(size(xgrid,1),1);
        for ii=1:length(pidcell1)
            c1 = pidcell1{ii}';
            xxx = xx(c1);
            xv = xxx(:);
            bb = sortrows(xv,[1]);
            lb = bb(1,:); ub = bb(end,:);
            bii = (xgrid(:,1)>=lb(1)) & (xgrid(:,1)<=ub(1));
            vv(bii) = 1;
            nn = sum(bii);
            arr = unique(vec(ll(bii)));
            if ~isempty(arr)
                index = cellfun(@(x) isequal(arr,x), pidcell, 'UniformOutput', 0);
                if sum(cell2mat(index))==0
                    pidcell{cc} = arr;
                    cc = cc+1;
                end
            end
        end
        ill = cellfun('length',pidcell);
        cc = cell2mat(pidcell(ill<2));
        pidcell = pidcell(ill>=2);
        pidcell = [pidcell; {unique(cc)}];
        pidcell = pidcell(~cellfun('isempty',pidcell));
    case 2
        %         nx = size(xgrid,1);
        %         xx = reshape(1:nx,nx_true(1),[]);
        %
        %         len_true_k = round(max([len_true_k 1]));
        %         [nx1,nx2] = size(xx);
        %
        %         ni1 = round(nx1/nblock);
        %         ni2 = round(nx2/nblock);
        %
        %         if ni1<=len_true_k*2
        %             ni1 = len_true_k*2+1;
        %         end
        %         step = max([min([ni1-len_true_k*2,ni1]),1]);
        %         ii = [1:step:nx1]';
        %         jj = [0:ni1-1];
        %         kk = repmat(ii,1,ni1)+repmat(jj,size(ii,1),1);
        %         kii = kk(:,end)<=nx1;
        %         kii(sum(kii)+1) = 1;
        %         kk = kk(kii,:);
        %         gs = size(kk,2);
        %         kk = kk';
        %         kk(kk>nx1) = [];
        %         pidcell1 = mat2tiles(kk,[1,gs]);
        %
        %         if ni2<=len_true_k*2
        %             ni2 = len_true_k*2+1;
        %         end
        %         step = max([min([ni2-len_true_k*2,ni2]),1]);
        %         ii = [1:step:nx2]';
        %         jj = [0:ni2-1];
        %         kk = repmat(ii,1,ni2)+repmat(jj,size(ii,1),1);
        %         kii = kk(:,end)<=nx2;
        %         kii(sum(kii)+1) = 1;
        %         kk = kk(kii,:);
        %         gs = size(kk,2);
        %         kk = kk';
        %         kk(kk>nx2) = [];
        %         pidcell2 = mat2tiles(kk,[1,gs]);
        %
        %         pidcell = cell(length(pidcell1)*length(pidcell2),1);
        %         cc = 1;
        %         for ii=1:length(pidcell1)
        %             for jj=1:length(pidcell2)
        %                 c1 = pidcell1{ii}';
        %                 c2 = pidcell2{jj}';
        %                 xxx = xx(c1,c2);
        %                 %                 cla, imagesc(xx)
        %                 %                 hold on
        %                 %                 rectangle('Position',[c1(1),c2(1),size(xxx,1),size(xxx,2)],...
        %                 %                     'Curvature',[0.2,0.4],...
        %                 %                     'EdgeColor', 'r',...
        %                 %                     'LineWidth', 1,...
        %                 %                     'LineStyle','-')
        %                 %                 drawnow,pause
        %                 pidcell{cc} = xxx(:);
        %                 cc = cc+1;
        %             end
        %         end
        nx_true = ceil(range(xgrid));
        nx = prod(nx_true);
        xx = reshape(1:nx,[nx_true(1),nx_true(2)]);
        xs = linspace(floor(min(xgrid(:,1))),ceil(max(xgrid(:,1))),nx_true(1));
        ys = linspace(floor(min(xgrid(:,2))),ceil(max(xgrid(:,2))),nx_true(2));
        [xa,ya] = ndgrid(xs,ys);
        xg = [xa(:),ya(:)];
        
        len_true_k = round(max([len_true_k 1]));
        [nx1,nx2] = size(xx);
        
        pidcell1 = get_pidcell_1d(len_true_k, nx1, nblock);
        pidcell2 = get_pidcell_1d(len_true_k, nx2, nblock);
        
        pidcell = cell(length(pidcell1)*length(pidcell2),1);
        mg = max_unit;%round(size(xgrid,1)/nblock);
        cc = 1;
        ll = 1:size(xgrid,1);
        vv = zeros(size(xgrid,1),1);
        for ii=1:length(pidcell1)
            for jj=1:length(pidcell2)
                c1 = pidcell1{ii}';
                c2 = pidcell2{jj}';
                xxx = xx(c1,c2);
                xv = xg(xxx(:),:);
                bb = sortrows(xv,[1,2]);
                lb = bb(1,:); ub = bb(end,:);
                bii = (xgrid(:,1)>=lb(1)) & (xgrid(:,1)<=ub(1)) & (xgrid(:,2)>=lb(2)) & (xgrid(:,2)<=ub(2));
                vv(bii) = 1;
                nn = sum(bii);
                if nn<=mg
                    arr = unique(vec(ll(bii)));
                    if ~isempty(arr)
                        index = cellfun(@(x) isequal(arr,x), pidcell, 'UniformOutput', 0);
                        if sum(cell2mat(index))==0
                            pidcell{cc} = arr;
                            cc = cc+1;
                        end
                    end
                end
                
                if nn>mg
                    fbii = find(bii==1);
                    xxg = xgrid(bii,:);
                    [xxgs,sii] = sortrows(xxg,[1,2]);
                    fbii = fbii(sii);
                    ng = mat2tiles(1:nn,[1,mg]);
                    for mm=1:length(ng)
                        arr = unique(vec(ll(fbii(ng{mm}))));
                        if ~isempty(arr)
                            index = cellfun(@(x) isequal(arr,x), pidcell, 'UniformOutput', 0);
                            if sum(cell2mat(index))==0
                                pidcell{cc} = arr;
                                cc = cc+1;
                            end
                        end
                    end
                end
                
            end
        end
        ill = cellfun('length',pidcell);
        cc = cell2mat(pidcell(ill<10));
        pidcell = pidcell(ill>=10);
        pidcell = [pidcell; {unique(cc)}];
        pidcell = pidcell(~cellfun('isempty',pidcell));
    case 3
        nx_true = ceil(range(xgrid));
        nx = prod(nx_true);
        xx = reshape(1:nx,[nx_true(1),nx_true(2),nx_true(3)]);
        xs = linspace(floor(min(xgrid(:,1))),ceil(max(xgrid(:,1))),nx_true(1));
        ys = linspace(floor(min(xgrid(:,2))),ceil(max(xgrid(:,2))),nx_true(2));
        zs = linspace(floor(min(xgrid(:,3))),ceil(max(xgrid(:,3))),nx_true(3));
        [xa,ya,za] = ndgrid(xs,ys,zs);
        xg = [xa(:),ya(:),za(:)];
        
        len_true_k = round(max([len_true_k 1]));
        [nx1,nx2,nx3] = size(xx);
        
        pidcell1 = get_pidcell_1d(len_true_k, nx1, nblock);
        pidcell2 = get_pidcell_1d(len_true_k, nx2, nblock);
        pidcell3 = get_pidcell_1d(len_true_k, nx3, nblock);
        
        pidcell = cell(length(pidcell1)*length(pidcell2)*length(pidcell3),1);
        mg = max_unit;%round(size(xgrid,1)/nblock);
        cc = 1;
        ll = 1:size(xgrid,1);
        vv = zeros(size(xgrid,1),1);
        for ii=1:length(pidcell1)
            for jj=1:length(pidcell2)
                for kk=1:length(pidcell3)
                    c1 = pidcell1{ii}';
                    c2 = pidcell2{jj}';
                    c3 = pidcell3{kk}';
                    xxx = xx(c1,c2,c3);
                    xv = xg(xxx(:),:);
                    bb = sortrows(xv,[1,2,3]);
                    lb = bb(1,:); ub = bb(end,:);
                    bii = (xgrid(:,1)>=lb(1)) & (xgrid(:,1)<=ub(1)) & (xgrid(:,2)>=lb(2)) & (xgrid(:,2)<=ub(2)) & (xgrid(:,3)>=lb(3)) & (xgrid(:,3)<=ub(3));
                    vv(bii) = 1;
                    nn = sum(bii);
                    if nn<=mg
                        arr = unique(vec(ll(bii)));
                        if ~isempty(arr)
                            index = cellfun(@(x) isequal(arr,x), pidcell, 'UniformOutput', 0);
                            if sum(cell2mat(index))==0
                                pidcell{cc} = arr;
                                cc = cc+1;
                            end
                        end
                    end
                    
                    if nn>mg
                        fbii = find(bii==1);
                        xxg = xgrid(bii,:);
                        [xxgs,sii] = sortrows(xxg,[1,2,3]);
                        fbii = fbii(sii);
                        ng = mat2tiles(1:nn,[1,mg]);
                        for mm=1:length(ng)
                            arr = unique(vec(ll(fbii(ng{mm}))));
                            if ~isempty(arr)
                                index = cellfun(@(x) isequal(arr,x), pidcell, 'UniformOutput', 0);
                                if sum(cell2mat(index))==0
                                    pidcell{cc} = arr;
                                    cc = cc+1;
                                end
                            end
                        end
                    end
                    
                end
            end
        end
        ill = cellfun('length',pidcell);
        cc = cell2mat(pidcell(ill<10));
        pidcell = pidcell(ill>=10);
        pidcell = [pidcell; {unique(cc)}];
        pidcell = pidcell(~cellfun('isempty',pidcell));
end




