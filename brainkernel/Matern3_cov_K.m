function [K, r2] = Matern3_cov_K(x,z,ell,sf2)
r2 = r2_xz_matern3(x,z,ell);

K = sf2*exp(-sqrt(3*r2)).*(1+sqrt(3*r2));
