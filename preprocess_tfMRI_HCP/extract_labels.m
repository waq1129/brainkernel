function labels1 = extract_labels(aa,start_t,ns1)
bb = reshape(aa',2,[])';
tt = bb(:,2);
tt1 = cellfun(@str2num, tt);
[ii,jj] = sort(tt1);
bb = bb(jj,:);

% start_t = 29314;
delay = 5500;
sr = 720;
cate = unique(bb(:,1));

tt = [];
cc = [];
for ii=1:size(bb,1)
    ss = bb(ii,2); ss = ss{1};
    tt = [tt; ceil((str2num(ss)-start_t+delay)/sr)];
    ss = bb(ii,1); ss = ss{1};
    mm = cellfun(@(x) strcmp(x,ss),cate);
    [jj,kk] = find(mm==1);
    cc = [cc; jj(1)];
end
labels = [cc,tt];

start_p = [labels(1,:) 1];
end_p = [];
for ii=2:size(bb,1)
    ll = labels(ii,:);
    if ll(1)~=start_p(end,1)
        start_p = [start_p; ll ii];
    end
end

ee = start_p(2:end,end)-1;
end_p = [labels(ee,:); labels(end,:)];
start_p = start_p(:,1:2);
se_p = [start_p end_p(:,2)];

%%
ind = [1:ns1]';
labels = zeros(ns1,1);
cc = 1;
for ii=1:ns1
    dd = ii-se_p(:,2)>=0 & se_p(:,3)-ii>=0;
    ee = find(dd==1);
    if ~isempty(ee)
        labels(ii) = se_p(ee,1);
    end
end
labels1 = labels;