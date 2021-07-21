Run generate_brainkernel.m to generate a brain kernel using the resting-state fMRI in HCP.  It takes a long time to run and requires 10G+ space to store intermediate variables. 

Inside generate_brainkernel.m, you first run the 4 steps to fit the brain kernel model:
brainkernel_estimate_step1.m
brainkernel_estimate_step2.m
brainkernel_estimate_step3.m
brainkernel_estimate_step4.m

After fitting the model, you should obtain a brainkernel_latent.mat file from step4. That contains the latent for the voxels in the HCP space. We need to generate a function based on the inferred latent for applying to task fMRIs.

So run brainkernel_pred_latent.m to generate KinvF=Kinv*(F-fmu) first, and then use KinvF to infer latent for new task voxels. The last cell in brainkernel_pred_latent.m tells you how to use KinvF to predict the latent for new voxels. Then you can use the new latent to fit an RBF kernel to the task data, i.e., replacing the 3D voxel coordinates with the new latent (20d).

We provide the latent in brainkernel_latent.mat and the KinvF in brainkernel_KinvF.mat for you. So that you don't need to fit the brain kernel to resting-state data yourself. Users can just download it and use it instead of the RBF covariance. 