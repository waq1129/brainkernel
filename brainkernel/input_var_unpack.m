function [pf,palpha,phyp,pgphyp,optid] = input_var_unpack(pp,input_var,nc,nf,nx)
optid = input_var.optid;
if optid(1) == 0
    pf = input_var.pf;
else
    pf = pp(1:nx*nf);
end
if optid(2) == 0
    palpha = input_var.palpha;
else
    palpha = pp(1+nx*nf*optid(1):nx*nf*optid(1)+nc*nf);
end

if optid(3) == 0
    phyp = input_var.phyp;
else
    phyp = pp(1+nx*nf*optid(1)+nc*nf*optid(2):nx*nf*optid(1)+nc*nf*optid(2)+3);
end

if optid(4) == 0
    pgphyp = input_var.pgphyp;
else
    pgphyp = pp(1+nx*nf*optid(1)+nc*nf*optid(2)+3*optid(3):nx*nf*optid(1)+nc*nf*optid(2)+3*optid(3)+1);
end
