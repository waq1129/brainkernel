function labels = read_label_from_csv(subid)
% subid = '102311';
nn = ['/Users/anqiwu/Downloads/HCP_WB_Tutorial_1.0/' subid '/MNINonLinear/Results/tfMRI_WM_RL/WM_run1_TAB.txt'];
data = textread(nn, '%s');
mm = find(cellfun(@(x) strcmp(x,'InitialTR'),data)==1);
sot = data(mm(1)+7);
sot1 = str2num(sot{1});

aa = [];

oo = 'Tools';
tt = find(cellfun(@(x) strcmp(x,oo),data)==1);
ttonset = data(tt+14);
ss = repmat({oo},length(tt),1);
aa = [aa; ss(:) ttonset(:)];

oo = 'Body';
tt = find(cellfun(@(x) strcmp(x,oo),data)==1);
ttonset = data(tt+14);
ss = repmat({oo},length(tt),1);
aa = [aa; ss(:) ttonset(:)];

oo = 'Face';
tt = find(cellfun(@(x) strcmp(x,oo),data)==1);
ttonset = data(tt+14);
ss = repmat({oo},length(tt),1);
aa = [aa; ss(:) ttonset(:)];

oo = 'Place';
tt = find(cellfun(@(x) strcmp(x,oo),data)==1);
ttonset = data(tt+14);
ss = repmat({oo},length(tt),1);
aa = [aa; ss(:) ttonset(:)];

aa1 = aa';
aa1 = aa1(:);

%%
nn = ['/Users/anqiwu/Downloads/HCP_WB_Tutorial_1.0/' subid '/MNINonLinear/Results/tfMRI_WM_LR/WM_run2_TAB.txt'];
data = textread(nn, '%s');
mm = find(cellfun(@(x) strcmp(x,'InitialTR'),data)==1);
sot = data(mm(1)+7);
sot2 = str2num(sot{1});

aa = [];

oo = 'Tools';
tt = find(cellfun(@(x) strcmp(x,oo),data)==1);
ttonset = data(tt+14);
ss = repmat({oo},length(tt),1);
aa = [aa; ss(:) ttonset(:)];

oo = 'Body';
tt = find(cellfun(@(x) strcmp(x,oo),data)==1);
ttonset = data(tt+14);
ss = repmat({oo},length(tt),1);
aa = [aa; ss(:) ttonset(:)];

oo = 'Face';
tt = find(cellfun(@(x) strcmp(x,oo),data)==1);
ttonset = data(tt+14);
ss = repmat({oo},length(tt),1);
aa = [aa; ss(:) ttonset(:)];

oo = 'Place';
tt = find(cellfun(@(x) strcmp(x,oo),data)==1);
ttonset = data(tt+14);
ss = repmat({oo},length(tt),1);
aa = [aa; ss(:) ttonset(:)];

aa2 = aa';
aa2 = aa2(:);

%%
if sot1<sot2
    labels1 = extract_labels(aa1,sot1,2400);
    labels2 = extract_labels(aa2,sot2,2400);
    labels = [labels1(1:405);labels2(1:405)];
else
    labels1 = extract_labels(aa2,sot2,2400);
    labels2 = extract_labels(aa1,sot1,2400);
    labels = [labels1(1:405);labels2(1:405)];
end

