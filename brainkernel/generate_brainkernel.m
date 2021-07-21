clc,clear,addpath(genpath(pwd)); warning off

% estimate brain kernel
display('step 1 ...')
brainkernel_estimate_step1

display('step 2 ...')
brainkernel_estimate_step2

display('step 3 ...')
brainkernel_estimate_step3

display('step 4 ...')
brainkernel_estimate_step4

% predict latent z in order to construct the brain for new voxels 
display('predict latent z in order to construct the brain for new voxels ...')
brainkernel_pred_latent
