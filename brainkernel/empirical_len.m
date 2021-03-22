function len_est = empirical_len(S,xpos,plot_flag)
Kernel = S;
Kernel(Kernel<0) = 0;
if plot_flag, subplot(431),imagesc(Kernel),colorbar,axis image, end
%%
condthresh = 1e10;
minlens = 1;
nc = size(xpos,2); % number of dimensions
minlens = repmat(minlens,nc,1);
xwid = range(xpos)'; % estimated support along each dimension
nxcirc = ceil(max([xwid(:)'+minlens(:)'*4; xwid(:)'*2]))';

% Make Fourier basis for each input dimension
Bmats = cell(nc,1); % Fourier basis matrix for each filter dimension
wwnrmvecs = cell(nc,1); % Fourier frequencies for each filter dimension
cdiagvecs = cell(nc,1); % eigenvalues for each dimension
for jj = 1:nc
    % Move to range [0 xwid(jj)].
    if min(xpos(:,jj))>0
        xpos(:,jj) = xpos(:,jj)-min(xpos(:,jj));
    end
    
    % determine maximal freq for Fourier representation
    maxfreq = floor(nxcirc(jj)/(pi*minlens(jj))*sqrt(.5*log(condthresh)));
    
    % Compute basis for non-uniform DFT and frequency vector
    [Bmats{jj},wvecs] = realnufftbasis(xpos(:,jj),nxcirc(jj),sum(nxcirc));%nxcirc(jj));
    wwnrmvecs{jj} = (2*pi/nxcirc(jj))^2*(wvecs.^2); % normalized freqs squared
    cdiagvecs{jj} = exp(-.5*wwnrmvecs{jj}*minlens(jj).^2); % diagonal of cov
end

wll = cellfun(@(a) length(a), wwnrmvecs);
wl = min(wll);
wdd = ceil(wl/2);
w1 = cellfun(@(a) a(1), wwnrmvecs);
wwnrms = wwnrmvecs{1}(1:wdd)+sum(w1(2:end));
Cdiag = exp(-.5*wwnrms*minlens(1).^2); % diagonal of cov

%% original kerne
Bfft_cut = column_kron_cut(Bmats,wdd);
ks = sum((Bfft_cut*Kernel).*Bfft_cut,2);
if plot_flag,
    subplot(432);plot(ks);
    subplot(435);plot(Cdiag);
end
%
ff = ks;
ff1 = ff;
llist = linspace(0.1,20,10);
ss = [];
for ll=llist
    [a,b,fff] = fit_gaussian(exp(-.5*wwnrms*ll.^2),ff);
    dd = (fff-ff)'*(fff-ff);
    ss = [ss; dd];
    if plot_flag, subplot(433),plot([fff ff]),drawnow, end
end
[s1,mm] = min(ss);
len_est1 = llist(mm);
if plot_flag, subplot(436),plot(ss),title(len_est1), end

%%
Cmat0 = Kernel;
c1 = Cmat0(:,1);
[cc,ii] = sort(c1);
Cmat1 = Cmat0(ii,ii);
if plot_flag,
    subplot(235);imagesc(Cmat0); colorbar, axis image, drawnow;
    subplot(236);imagesc(Cmat1); colorbar, axis image, drawnow;
end
%
ks = sum((Bfft_cut*Cmat1).*Bfft_cut,2);
if plot_flag,
    subplot(437);plot(ks);
    subplot(4,3,10);plot(Cdiag);
end
%
ff = ks;
ff2 = ff;
llist = linspace(0.1,20,10);
ss = [];
for ll=llist
    %     ll
    [a,b,fff] = fit_gaussian(exp(-.5*wwnrms*ll.^2),ff);
    dd = (fff-ff)'*(fff-ff);
    ss = [ss; dd];
    if plot_flag, subplot(438),plot([fff ff]),drawnow, end
end
[s2,mm] = min(ss);
len_est2 = llist(mm);
if plot_flag, subplot(4,3,11),plot(ss),title(len_est2), end

%%
if 1%s1<s2
    len_est = len_est1;
    ff = ff1;
else
    len_est = len_est2;
    ff = ff2;
end
[a,b,fff] = fit_gaussian(exp(-.5*wwnrms*len_est.^2),ff);
[a,b,cd] = fit_gaussian(Cdiag,fff);

if plot_flag, subplot(439),plot([fff ff cd]),drawnow, end
if minlens(1)>len_est
    len_est = minlens(1);
end
disp(['len_est: ' num2str(len_est)])

if plot_flag, subplot(4,3,12),imagesc(covSEiso([log(len_est);0], xpos)),colorbar,axis image,drawnow, end

