Edited from VITON Readme:

Zalando dataset: [Google Drive](https://drive.google.com/drive/folders/1-RIcmjQKTqsf3PZsoHT4hivNngx_3386?usp=sharing).

Put all folder and labels in the ```data``` folder:

```data/women_top```: reference images (image name is ID_0.jpg) and clothing images (image name is ID_1.jpg). For example, the clothing image on reference image 000001_0.jpg is 000001_1.jpg. The resolution of these images is 1100x762.

```data/pose.pkl```: a pickle file containing a dictionary of the pose keypoints of each reference image. We used https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation to extract Pose keypoints. 

NOTE : Use the demo.py from this REPO to save pose keypoints as pickle file. instead of original demo.ipynb

```data/segment```: folder containing the segmentation map of each reference image. Used LIPP_JPPNET for that. 



### Test

#### First stage
Download pretrained models on [Google Drive](https://drive.google.com/drive/folders/1qFU4KmvnEr4CwEFXQZS_6Ebw5dPJAE21?usp=sharing). Put them under ```model/``` folder.

Run ```test_stage1.sh``` to do the inference.
The results are in ```results/stage1/images/```. ```results/stage1/index.html``` visualizes the results.

#### Second stage

Run the matlab script ```shape_context_warp.m``` to extract the TPS transformation control points.

Then ```test_stage2.sh``` will do the refinement and generate the final results, which locates in ```results/stage2/images/```. ```results/stage2/index.html``` visualizes the results.


### Train

#### Prepare data
Go inside ```prepare_data```. 

First run ```extract_tps.m```. This will take sometime, you can try run it in parallel or directly download the pre-computed TPS control points via Google Drive and put them in ```data/tps/```.

Then run ```./preprocess_viton.sh```, and the generated TF records will be in ```prepare_data/tfrecord```.


#### First stage
Run ```train_stage1.sh```

#### Second stage
Run ```train_stage2.sh```


