clc,clear,addpath(genpath(pwd)); warning off

% preprocess for generating brain kernel
display('preprocessing step ...')
brainkernel_preprocess();

% estimate brain kernel
display('start to estimate ...')
brainkernel_estimate();
