function len = bound_len(pgphyp,minl,maxl)
len = exp(-pgphyp/2);
len = min([len,maxl]);
len = max([len,minl]);
len = -2*log(len);