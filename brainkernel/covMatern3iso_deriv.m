function [A,dA] = covMatern3iso_deriv(loghyper, x, z, a, test)

% Squared Exponential covariance function with isotropic distance measure. The
% covariance function is parameterized as:
%
% k(x^p,x^q) = sf2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2)
%
% where the P matrix is ell^2 times the unit matrix and sf2 is the signal
% variance. The hyperparameters are:
%
% loghyper = [ log(ell)
%              log(sqrt(sf2)) ]
%
% For more help on design of covariance functions, try "help covFunctions".
%
% (C) Copyright 2006 by Carl Edward Rasmussen (2007-06-25)

if nargin == 0, A = '2'; return; end              % report number of parameters
if nargin == 2, z = x; end
if nargin<4
    a = 1;
    test = 1;
end
[n D] = size(x);
ell = exp(loghyper(1));                           % characteristic length scale
sf2 = exp(2*loghyper(2));                                     % signal variance

% A = sf2*exp(-sq_dist(x'/ell,z'/ell)/2);
[A, r2] = Matern3_cov_K(x,z,ell,sf2);

if nargout>1
    nx = size(x,1);
    nz = size(z,1);
    
    x1 = vec(x);
    x2 = repmat(x1,1,nz)';
    x3 = reshape(x2,[nz,nx,D]);
    x4 = permute(x3,[1,3,2]);
    x5 = reshape(x4,nz,[]);
    x6 = reshape(x5,1,[]);
    x7 = reshape(x6',D*nz,[])';
    
    z1 = vec(z);
    z2 = repmat(z1,1,nx)';
    
    dd = x7-z2;
    
    dA1 = sqrt(3)*A.*r2.^(-0.5);
    dA2 = sf2*sqrt(3)*exp(-sqrt(3*r2)).*r2.^(-0.5);
    dA = repmat(dA1,1,D).*dd/ell^2-repmat(dA2,1,D).*dd/ell^2;
    
    bb = reshape(dA,[],D)';
    cc = reshape(bb,[D,nx,nz]);
    dd = permute(cc,[1,3,2]);
    ee = reshape(dd,D,[]);
    ff = reshape(ee,D*nz,[])';
    
    dA = ff;
    
end