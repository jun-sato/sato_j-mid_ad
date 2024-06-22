# Annotation-free multi-organ anomaly detection in abdominal CT using free-text radiology reports

## Features

Train an image classification model by extracting labels from radiology free-text reports.  
This reduces the labor-intensive annotation work and allows the creation of a large dataset.  
A multi-organ segmentation model and an information extraction schema were used to extract specific organ images and disease information, CT images and radiology reports, respectively, which were used to train a multiple-instance learning model for anomaly detection.

## Requirements

Required Libraries:

* monai
* SimpleITK
* Pytorch
* Timm
* matplotlib

## Installation

Here is how to install the listed libraries.  
For training nnUNet, refer to the [official GitHub](https://github.com/MIC-DKFZ/nnUNet).  
Install PyTorch suitable for your GPU environment from the [official site](https://pytorch.org/get-started/locally/). 

If you are using conda, [recreating the environment with conda](https://qiita.com/nshinya/items/cb1cffabc3305c907bc5) is convenient.

```bash
conda env create -f=environment.yml
```


## model weights
The pretrained weights used in this study (Organ segmentation, anomaly detection, dataset size experiment) are available from [google drive](https://drive.google.com/drive/folders/17DgUVCo1We4EHM6PSeJ4ChtNZU5zhAJV?usp=sharing)



```
data  
 |-----liver
 |        |---jmid_0000000_0000.nii.gz  
 |        |---jmid_0000001_0000.nii.gz  
 |  
 |-----pancreas  
 |        |---jmid_0000000_0000.nii.gz  
 |        |---jmid_0000001_0000.nii.gz    
 |-----spleen
 |        |---jmid_0000000_0000.nii.gz  
 |        |---jmid_0000001_0000.nii.gz  
 |
```
### Abnormality Labels
```
Liver: [‘cyst’, ‘fatty_liver’, ‘bile_duct_dilation’, ‘SOL’, ‘deformation’, ‘calcification’, ‘pneumobilia’, ‘other_abnormality’, ‘nofinding’]
Gallbladder: [‘SOL’, ‘enlargement’, ‘deformation’, ‘gallstone’, ‘wall_thickening’, ‘polyp’, ‘other_abnormality’, ‘nofinding’]
Pancreas: [‘cyst’, ‘SOL’, ‘enlargement’, ‘atrophy’, ‘calcification’, ‘pancreatic_duct_dilation/atrophy’, ‘other_abnormality’, ‘nofinding’]
Spleen: [‘cyst’, ‘SOL’, ‘deformation’, ‘calcification’, ‘other_abnormality’, ‘nofinding’]
Kidney: [‘cyst’, ‘SOL(including_complicated_cyst)’, ‘enlargement’, ‘atrophy’, ‘deformation’, ‘calcification’, ‘other_abnormality’, ‘nofinding’]
Adrenal_gland: [‘SOL’, ‘enlargement’, ‘fat’, ‘calcification’, ‘other_abnormality’, ‘nofinding’]
Esophagus: [‘mass’, ‘hernia’, ‘dilation’, ‘other_abnormality’, ‘nofinding’]
```


 
# Usage
## Script for model training/inference
### training_25D_multi.py
A file for training an anomaly detection model using images of excised organs.
Using PyTorch DataParallel.

```
python training_25D_multi.py --batch_size 16 --datadir /path/to/the/data/directory --num_epochs 50 --organ liver --save_model_name /model/path 
```

## evaluation_25D.py
Evaluation of predictions and accuracy using the trained model.

```
python evaluation_25D.py --datadir /path/to/the/data/directory --organ liver --weight_path /model/path --segtype 25D --outputdir ../result_eval/
```

Output files:  
  - AUC curve and Confusion matrix graphs are created on the output directory.
 
 
# Author information
 
* Author: Junya Sato
* 2022/8/21 initial commit  

# License

### Code

The code in this repository is licensed under the Apache License 2.0. You are free to use, modify, and distribute it, provided that you comply with the terms of the license.

### Pretrained Model Weights

The pre-trained model weights are licensed under the Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International (CC BY-NC-ND 4.0) license. You may use and share the model weights for non-commercial purposes only, and you may not distribute modified versions of the weights.

[Link to model weights on Google Drive](https://drive.google.com/drive/folders/17DgUVCo1We4EHM6PSeJ4ChtNZU5zhAJV?usp=sharing)





