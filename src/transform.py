import albumentations
from monai.transforms import (
    OneOf,
    RandAdjustContrastd,
    RandScaleIntensityd,
    RandGaussianNoised,
    EnsureChannelFirstd,
    EnsureChannelFirst,
    AsDiscrete,
    Activations,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    Orientation,
    RandCropByPosNegLabeld,
    RandRotated,
    ScaleIntensityRanged,
    ScaleIntensityRange,
    Pad,
    MaskIntensityd,
    Spacingd,
    SpatialPadd,
    Resize,
    RandRotate90d,
    RandAffined,
    RandFlipd,
    SpatialPad,
    RandFlip,
    RandRotate90,
    RandShiftIntensity
)
def get_transforms(image_size,seed):
    train_transforms = Compose([
        RandRotate90d(keys=['image'], prob=0.5, spatial_axes=(0, 1)),
        RandFlipd(keys=['image'], spatial_axis=[0,1], prob=0.5), # ランダムフリップを追加
    ])


    val_transforms = albumentations.Compose([
    albumentations.Resize(image_size[0], image_size[1]),])

    return train_transforms,val_transforms


def get_val_transform(image_size):
    val_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-200,
                a_max=400,
                b_min=-1.0,
                b_max=1.0,
                clip=True,
            ),
            #SpatialPadd(keys=["image"],spatial_size=(256, 256, 64)),
            Resized(keys=["image"], spatial_size=input_size),
        ]
    )
    return val_transforms


def get_nifti_transform():
    train_transforms = Compose(
        [
            ScaleIntensityRange(
                a_min=-200,
                a_max=400,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            SpatialPad(spatial_size=[64, 256, 256]),
            Resize(spatial_size=(-1,256, 256)),
            RandFlip(
                spatial_axis=[0],
                prob=0.10,
            ),        
            RandFlip(
                spatial_axis=[1],
                prob=0.10,
            ),
            RandFlip(
                spatial_axis=[2],
                prob=0.10,
            ),
            RandShiftIntensity(
                offsets=0.10,
                prob=0.50,
            ),
        ]
    )
    val_transforms = Compose(
        [
            ScaleIntensityRange(
                a_min=-200,
                a_max=400,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            SpatialPad(spatial_size=[64, 256, 256]),
            Resize(spatial_size=(-1,256, 256)),
        ]
    )
    return train_transforms,val_transforms