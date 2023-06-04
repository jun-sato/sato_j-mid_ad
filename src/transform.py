import albumentations
# from monai.transforms import (
#     OneOf,
#     RandAdjustContrastd,
#     RandScaleIntensityd,
#     RandGaussianNoised,
#     EnsureChannelFirstd,
#     AsDiscrete,
#     Activations,
#     Compose,
#     CropForegroundd,
#     LoadImaged,
#     Orientationd,
#     RandCropByPosNegLabeld,
#     RandRotated,
#     ScaleIntensityRanged,
#     MaskIntensityd,
#     Spacingd,
#     SpatialPadd,
#     Resized
# )
def get_transforms(image_size,seed):
    # Define transforms
    train_transforms = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        #albumentations.HorizontalFlip(p=0.5),
        #albumentations.VerticalFlip(p=0.5),
        #albumentations.Transpose(p=0.5),
        #albumentations.RandomBrightness(limit=0.1, p=0.7),
        #albumentations.ShiftScaleRotate(shift_limit=0.3, scale_limit=0.3, rotate_limit=45, border_mode=4, p=0.7),

        #albumentations.OneOf([
        #    albumentations.MotionBlur(blur_limit=3),
        #    albumentations.MedianBlur(blur_limit=3),
        #    albumentations.GaussianBlur(blur_limit=3),
        #    albumentations.GaussNoise(var_limit=(3.0, 9.0)),
        #], p=0.5),
        #albumentations.OneOf([
        #    albumentations.OpticalDistortion(distort_limit=1.),
        #    albumentations.GridDistortion(num_steps=5, distort_limit=1.),
        #], p=0.5),

        #albumentations.Cutout(max_h_size=int(image_size * 0.5), max_w_size=int(image_size * 0.5), num_holes=1, p=0.5),
    ])

    val_transforms = albumentations.Compose([
    albumentations.Resize(image_size, image_size),])
    return train_transforms,val_transforms
