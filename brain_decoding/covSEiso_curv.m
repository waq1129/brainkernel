function [K, blockK] = covSEiso_curv(loghyper, x, z,curvature,test,rescale)

% Matern covariance function with nu = 5/2 and isotropic distance measure. The
% covariance function is:
%
% k(x^p,x^q) = s2f * (1 + sqrt(5)*d + 5*d/3) * exp(-sqrt(5)*d)
%
% where d is the distance sqrt((x^p-x^q)'*inv(P)*(x^p-x^q)), P is ell times
% the unit matrix and sf2 is the signal variance. The hyperparameters are:
%
% loghyper = [ log(ell)
%              log(sqrt(sf2)) ]
%
% For more help on design of covariance functions, try "help covFunctions".
%
% (C) Copyright 2006 by Carl Edward Rasmussen (2006-03-24)

if nargin == 0, A = '2'; return; end
if nargin<3, z = []; end
if nargin<4, curvature = 0; end
if nargin<5, test = 0; end
if nargin<6, rescale = 0; end

if ~test, z = x; end
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;        % determine mode

[n, D] = size(x);
loghyper(1) = max(loghyper(1),-10);
ell = exp(loghyper(1));
sf2 = exp(2*loghyper(2));
if rescale
    x = x/ell; z = z/ell; ell = 1;
end
K = SE_cov_K(x,z,ell,sf2);

if nargout >= 2
    if curvature==0
        blockK = K;
    end
    
    if curvature==1
        if test
            D = size(x,2);
            Ki = [];
            for ii=1:D
                dK_r2 = SE_cov_dK(x,z,ell,sf2,[ii,0])';
                Ki = [Ki dK_r2];
            end
            
            blockK = [K' Ki]';
        else
            if xeqz
                z = x;
            end
            D = size(x,2);
            Ki = [];
            for ii=1:D
                dK_r2 = SE_cov_dK(x,z,ell,sf2,[0,ii]);
                Ki = [Ki dK_r2];
            end
            blockK = [K Ki];
        end
    end
    
    if curvature==2
        if test
            D = size(x,2);
            K1 = [];
            for ii=1:D
                dK_r2 = SE_cov_dK(x,z,ell,sf2,[ii,0])';
                K1 = [K1 dK_r2];
            end
            
            %%%%%%%%%%%%
            KK = [];
            for ii=1:D
                Kii = SE_cov_ddK(x,z,ell,sf2,[ii,0],[ii,0])';
                KK = [KK Kii];
            end
            K21 = KK;
            
            KK = [];
            for ii=1:D
                for jj=ii+1:D
                    Kii = SE_cov_ddK(x,z,ell,sf2,[0,ii],[0,jj])';
                    KK = [KK Kii];
                end
            end
            K22 = KK;
            
            K2 = [K21 K22];
            
            blockK = [K' K1 K2]';
            
        else
            if xeqz
                z = x;
            end
            
            nn = size(x,1);
            D = size(x,2);
            Ki = [];
            for ii=1:D
                dK_r2 = SE_cov_dK(x,z,ell,sf2,[0,ii]);
                Ki = [Ki dK_r2];
            end
            
            KK = [];
            for ii=1:D
                Kii = SE_cov_ddK(x,z,ell,sf2,[ii,0],[0,ii]);
                for jj=ii+1:D
                    Kij = SE_cov_ddK(x,z,ell,sf2,[ii,0],[0,jj]);
                    Kii = [Kii Kij];
                end
                Kii = [zeros(size(K,1), size(K,2)*(ii-1)) Kii];
                KK = [KK; Kii];
            end
            
            onesmask = kron(eye(D),ones(size(K)));
            KK1 = KK+KK';
            KK1(logical(onesmask)) = KK(logical(onesmask));
            
            K1 = [K Ki; Ki' KK1];
            
            %%%%%%%%%%%%
            KK = [];
            KK1 = zeros(D*nn,D*nn);
            for ii=1:D
                Kii = SE_cov_ddK(x,z,ell,sf2,[ii,0],[ii,0])';
                KK = [KK Kii];
                KK1((ii-1)*nn+1:ii*nn,(ii-1)*nn+1:ii*nn) = SE_cov_dddK(x,z,ell,sf2,[ii,0],[0,ii],[0,ii]);
                for jj=1:D
                    if jj==ii
                        continue;
                    end
                    KK1((ii-1)*nn+1:ii*nn,(jj-1)*nn+1:jj*nn) = SE_cov_dddK(x,z,ell,sf2,[0,jj],[0,jj],[ii,0]);
                end
            end
            K21 = [KK; KK1];
            
            [r1, c1, pp] = gen_rc(D, eye(D));
            
            KK = [];
            for ii=1:D
                for jj=ii+1:D
                    Kii = SE_cov_ddK(x,z,ell,sf2,[0,ii],[0,jj]);
                    KK = [KK Kii];
                end
            end
            
            KK1 = [];
            for ii=1:D
                KK11 = [];
                for jj=1:D*(D-1)/2
                    
                    if c1(jj)==ii
                        KK11 = [KK11 SE_cov_dddK(x,z,ell,sf2,[0,ii],[0,ii],[r1(jj),0])];
                    end
                    if r1(jj)==ii
                        KK11 = [KK11 SE_cov_dddK(x,z,ell,sf2,[0,ii],[0,ii],[c1(jj),0])];
                    end
                    if c1(jj)~=ii && r1(jj)~=ii
                        KK11 = [KK11 SE_cov_dddK(x,z,ell,sf2,[0,ii],[0,c1(jj)],[r1(jj),0])];
                    end
                    
                end
                KK1 = [KK1; KK11];
            end
            K22 = [KK; KK1];
            
            K2 = [K21 K22];
            
            %%%%%%%%%%%%%
            KK1 = zeros(D*nn,D*nn);
            for ii=1:D
                KK1((ii-1)*nn+1:ii*nn,(ii-1)*nn+1:ii*nn) = SE_cov_ddddK(x,z,ell,sf2,[ii,0],[ii,0],[0,ii],[0,ii]);
                for jj=1:D
                    if jj==ii
                        continue;
                    end
                    KK1((ii-1)*nn+1:ii*nn,(jj-1)*nn+1:jj*nn) = SE_cov_ddddK(x,z,ell,sf2,[jj,0],[jj,0],[0,ii],[0,ii]);
                end
            end
            
            
            KK2 = [];
            for ii=1:D
                KK21 = [];
                for jj=1:D*(D-1)/2
                    
                    if c1(jj)==ii
                        KK21 = [KK21 SE_cov_ddddK(x,z,ell,sf2,[ii,0],[ii,0],[0,ii],[0,r1(jj)])];
                    end
                    if r1(jj)==ii
                        KK21 = [KK21 SE_cov_ddddK(x,z,ell,sf2,[ii,0],[ii,0],[0,ii],[0,c1(jj)])];
                    end
                    if c1(jj)~=ii && r1(jj)~=ii
                        KK21 = [KK21 SE_cov_ddddK(x,z,ell,sf2,[ii,0],[0,ii],[0,r1(jj)],[c1(jj),0])];
                    end
                    
                end
                KK2 = [KK2; KK21];
            end
            
            
            KK3 = [];
            for ii=1:D*(D-1)/2
                KK31 = [];
                for jj=1:D*(D-1)/2
                    ci = c1(ii);
                    ri = r1(ii);
                    cj = c1(jj);
                    rj = r1(jj);
                    
                    if ii==jj
                        KK31 = [KK31 SE_cov_ddddK(x,z,ell,sf2,[ri,0],[ri,0],[ci,0],[ci,0])];
                    else
                        if ci==cj
                            KK31 = [KK31 SE_cov_ddddK(x,z,ell,sf2,[0,ri],[rj,0],[ci,0],[0,cj])];
                        else
                            if ri==cj
                                KK31 = [KK31 SE_cov_ddddK(x,z,ell,sf2,[0,ci],[rj,0],[ri,0],[0,cj])];
                            else
                                if ri==rj
                                    KK31 = [KK31 SE_cov_ddddK(x,z,ell,sf2,[0,ci],[cj,0],[ri,0],[0,rj])];
                                else
                                    if ci==rj
                                        KK31 = [KK31 SE_cov_ddddK(x,z,ell,sf2,[0,ri],[cj,0],[ci,0],[0,rj])];
                                    else
                                        KK31 = [KK31 SE_cov_ddddK(x,z,ell,sf2,[ri,0],[cj,0],[ci,0],[rj,0])];
                                    end
                                end
                            end
                        end
                        
                    end
                end
                KK3 = [KK3; KK31];
            end
            
            K3 = [KK1 KK2; KK2' KK3];
            
            blockK = [K1 K2; K2' K3];
        end
    end
end

end
