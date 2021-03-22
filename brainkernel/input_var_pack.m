function [pp,input_var] = input_var_pack(pf,palpha,phyp,pgphyp,optid)
pp = [];
if optid(1) == 0
    input_var.pf = pf;
else
    input_var.pf = [];
    pp = [pp; vec(pf)];
end
if optid(2) == 0
    input_var.palpha = palpha;
else
    input_var.palpha = [];
    pp = [pp; vec(palpha)];
end
if optid(3) == 0
    input_var.phyp = phyp;
else
    input_var.phyp = [];
    pp = [pp; vec(phyp)];
end
if optid(4) == 0
    input_var.pgphyp = pgphyp;
else
    input_var.pgphyp = [];
    pp = [pp; vec(pgphyp)];
end
input_var.optid = optid;


