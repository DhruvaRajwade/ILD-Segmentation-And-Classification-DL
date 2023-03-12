ILD_Segmentation_Classification
==============================

This Repository contains code of our work on Segmenting Lung CT Images and classifying Interstitial Lung Diseases Using Deep Learning, submitted to the Respirology journal.

## Folders:
	Lung_Classification: Contains code for lung disease classification.
	Lung_Segmentatopn: Contains code for lung segmentation.
	Figures: Plots generated fr the paper
	example_notebooks: Contains example notebooks for both above tasks, which can be downloaded and run with user-data.

## Files:
	1. process_image.py: Augment the images and mask for the training dataset.
	2. data_generator.py: Dataset generator for the keras.
	3. infer.py: Run your model on test dataset and all the result are saved in the result` folder. The images are in the sequence: Image,Ground Truth Mask, Predicted Mask.
	4. run.py: Train the unet.
	5. unet.py: Contains the code for building the UNet architecture.
	6. resunet.py: Contains the code for building the ResUNet architecture.
	7. m_resunet.py: Contains the code for building the ResUNet++ architecture.
	8. mertrics.py: Contains the code for dice coefficient metric and dice coefficient loss. 

## Convert DICOM images to Jpeg images and numpy arrays using the dicom_utils.py file.

 ```python
 get_names(path)
 for name in names:
     final_image = convert_dcm_jpg(cdir, name)
     final_image.save(out_dir + '/' + name + '.jpg')
  ```
## Lung Segmentation Results For Models implemented

### 1. Performance on Geneva HRCT Dataset (12 classes, 512 x 512)

Training: 256x256 reshaped
validation: 256x256 reshaped

| Model        | Batch_Size | mIOU   | DSC    | Accuracy | Google Drive                                |  
|--------------|------------|--------|--------|----------|---------------------------------------------|
| UNET         | 10         | 0.8280 | 0.8604 | 0.9848   | [Link text Here](https://link-url-here.org) |   
| Attn_UNET    | 10         | 0.8394 | 0.8708 | 0.9901   | [Link text Here](https://link-url-here.org) |   
| R2_UNET      | 10         | **0.9191** | 0.8811 |**0.9934**   | [Link text Here](https://link-url-here.org) |   
| Attn_R2_UNET | 10         | 0.8523 | **0.9263** | 0.9913   | [Link text Here](https://link-url-here.org) |  
|              |            |        |        |          |                                             |  

#### Segmentation Results on Pascal VOC2012 (DeepLabv3Plus-MobileNet)

<div>
<img src="Figures/prediction1.png"   width="100%">
</div>

<div>
<img src="Figures/prediction2.png"   width="100%">
</div>

<div>
<img src="Figures/prediction3.png"   width="100%">
</div>

--------

