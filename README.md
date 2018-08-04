Readme of DeShadowNet

___________________________________

Important Files

I. train.py

train functions-
	local_s_train()- For training SNet, used to extract semantic contextual features.
	local_a_train()- For training ANet, used to do features which are more relevant for appearance modelling.

II. inference.py

Used for finding the inferences using GNet and reconstruct the image using shadow matte.
Steps to follow for training- 
	
	1. Keep the original images in folder ‘data/original_image’ folder, and keep the shadow removed images in folder ‘data/sr_image’. Till now, I am using a rubics to keep the original images and their shadow removed version with the same name but in different folder.
	2. Once, the data is prepared we can first call local_s_train and get a trained model using S-net out of it then we can use the output of that for training in A-net. The trained model after this can be used for inference using G-Net.

Steps to follow for inference-
	
	1. Run inference.py to reconstruct the image using the shadow matte created by the equation 1 in paper.
