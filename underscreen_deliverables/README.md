# Deliverables for HQ of flare removal and USC Face SR solutions.

There are following scripts:

* **downscale_flare_removal_train.py** script for training of Deep Feature solution downscale part.
* **fullres_flare_removal_train.py** script for training of Deep Feature solution fullres part.
* **flare_removal_fullres_inference.py** Deep feature flare removal inference script
* **sharpening_train.py** - train script of SR solution (can be used for faces and background)
* **sharpening_inference.py** - inference script of SR solution

Weights files _197cf30558a5b15cb3c7b76eb3918a3ffcca480e_2849.hdf5_ and _eb8101497b3d33692169d7feea669e28486e437a_4549.hdf5_
are for SR (different degradation levels),
while _20210303_dilation_stack_activations_shift.hdf5_ and _20210317_full_res_shift_optimized.hdf5_
are for Deep feature flare removal downscale and fullres parts accordingly.



