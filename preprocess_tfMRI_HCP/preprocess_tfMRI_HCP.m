% preprocess task fMRI from HCP. The demo is for the working memory task.
% Data is stored in the path with a format: subjectid/lr_WM_left.metric

clc,clear,addpath(genpath(pwd)); warning off

subjectid = '100307';
subjectid

metric_files = {'lr_WM', 'rl_WM'};
hems = {'left', 'right'};

bold = cell(2,1);
bold{1} = zeros(32492, 1200*4, 'single');
bold{2} = zeros(32492, 1200*4, 'single');


for hem = 1:2
    for i = 1:length(metric_files)
        fid = fopen(fullfile([num2str(subjectid) '/'],[metric_files{i} '_' hems{hem} '.metric']));
        xmlstr = fread(fid,'*char')';
        fclose(fid);
        block_starts = strfind(xmlstr, '<Data>');
        block_starts = block_starts + 6;
        block_ends = strfind(xmlstr, '</Data>');
        block_ends = block_ends - 1;
        ss = 1200;
        for t = 1:length(block_starts)
            bold{hem}(:,ss*(i-1) + t) = typecast(dunzip(base64decode( ...
                xmlstr(block_starts(t):block_ends(t)))), 'single');
        end
        bold{hem}(:,(ss*(i-1)+1):(ss*i)) = bsxfun(@minus, bold{hem}(:,(ss*(i-1)+1):(ss*i)), mean(bold{hem}(:,(ss*(i-1)+1):(ss*i)),2));
    end
end

%%
gg = load('gray');
orig_ind = gg.orig_ind;
coords = gg.coords;

wholebrain1 = [bold{1}(orig_ind{1},1:405); bold{2}(orig_ind{2},1:405)];
wholebrain1 = wholebrain1-repmat(mean(wholebrain1,2),1,size(wholebrain1,2));
wholebrain2 = [bold{1}(orig_ind{1},1201:1605); bold{2}(orig_ind{2},1201:1605)];
wholebrain2 = wholebrain2-repmat(mean(wholebrain2,2),1,size(wholebrain2,2));
wholebrain = [wholebrain1 wholebrain2];

%%
labels = read_label_from_csv(num2str(subjectid));

%%
save(['../HCP_data/hcp_sub' num2str(subjectid) '_WM'],'wholebrain','coords','orig_ind','labels')























