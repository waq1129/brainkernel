function [Bmats,Cdiag,cdiagvecs,wwnrm,wwnrmvecs] = myfft_nu(xpos,len, rho, minlen, maxlen, large_flag)
if nargin<5
    large_flag = 0;
    if nargin<4
        maxlen = minlen;
    end
end
condthresh = 1e12;
% Determine size of stimulus and its dimensions
nd = length(xpos); % number of dimensions
if length(minlen) == 1 % make vector out of len, if necessary
    minlen = repmat(minlen,nd,1);
end
if length(maxlen) == 1 % make vector out of len, if necessary
    maxlen = repmat(maxlen,nd,1);
end
if length(len) == 1 % make vector out of len, if necessary
    len = repmat(len,nd,1);
end
if length(rho) == 1 % make vector out of len, if necessary
    rho = repmat(rho,nd,1);
end

xwid = vec(cellfun(@(a) range(a), xpos)); % estimated support along each dimension
% nx = vec(cellfun(@(a) length(a), xpos));
% ii = xwid-nx>0;
% xwid(ii) = xwid(ii);
% xwid(~ii) = nx(~ii);

% Set nxcirc to default value if necessary
% nxcirc = ceil(max([xwid(:)'+maxlen(:)'*2; ...
%     xwid(:)'*1.25]))';
nxcirc = ceil(xwid(:)'*1.25)';
% nxcirc

% Make Fourier basis for each input dimension
Bmats = cell(nd,1); % Fourier basis matrix for each filter dimension
wwnrmvecs = cell(nd,1); % Fourier frequencies for each filter dimension
cdiagvecs = cell(nd,1); % eigenvalues for each dimension
condthresh_jj = (1e3)^(1/nd);
for jj = 1:nd
    % Move to range [0 xwid(jj)].
    if min(xpos{jj})>0
        xpos{jj} = xpos{jj}-min(xpos{jj});
    end
    
    % determine maximal freq for Fourier representation
    maxfreq = floor(nxcirc(jj)/(pi*minlen(jj))*sqrt(.5*log(condthresh)));
    % Compute basis for non-uniform DFT and frequency vector
    if large_flag
        [Bmats{jj},wvecs] = realnufftbasis(xpos{jj},nxcirc(jj),nxcirc(jj));%maxfreq*2+1);%
    else
        [Bmats{jj},wvecs] = realnufftbasis(xpos{jj},nxcirc(jj),maxfreq*2+1);%nxcirc(jj));%
    end
    wwnrmvecs{jj} = (2*pi/nxcirc(jj))^2*(wvecs.^2); % normalized freqs squared
    cdiagvecs{jj} = sqrt(2*pi)*rho(jj)*len(jj)*exp(-.5*wwnrmvecs{jj}*len(jj).^2); % diagonal of cov
    c_minlen = sqrt(2*pi)*rho(jj)*minlen(jj)*exp(-.5*wwnrmvecs{jj}*minlen(jj).^2); % diagonal of cov
    if large_flag
        ii = c_minlen>1/condthresh_jj;%-inf;%1e-1;%
    else
        ii = c_minlen>-inf;%1/condthresh_jj;%1e-1;%
    end
    cdiagvecs{jj} = cdiagvecs{jj}(ii);
    wwnrmvecs{jj} = wwnrmvecs{jj}(ii);
    Bmats{jj} = Bmats{jj}(ii,:);
end

Cdiag = 1;
wwnrm = 0;
for ii=1:nd
    Cdiag = kron(cdiagvecs{ii},Cdiag);
    wwnrm = tensorsum(wwnrmvecs{ii},wwnrm);
end