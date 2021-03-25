import imageio
import imgaug as ia
import cv2
import imgaug.augmenters as iaa

def get_preview(images, augmentationList):
    """
    Accepts a list of images and augmentationList as input.
    Provides a list of augmented images in that order as ouptut.
    """
    augmented = []
    for image in images:
        for augmentation in augmentationList:
            aug_id = augmentation['id']
            params = augmentation['params']
            if(aug_id==1):
                image = iaa.SaltAndPepper(p=params[0], per_channel=params[1])(image=image)
            elif(aug_id==2):
                image = iaa.imgcorruptlike.GaussianNoise(severity=(params[0],params[1]))(image=image)
            elif(aug_id==3):
                image = iaa.Rain(speed=(params[0],params[1]),drop_size=(params[2],params[3]))(image=image)
            elif(aug_id==4):
                image = iaa.imgcorruptlike.Fog(severity=(params[0],params[1]))(image=image)
            elif(aug_id==5):
                image = iaa.imgcorruptlike.Snow(severity=(params[0],params[1]))(image=image)
            elif(aug_id==6):
                image = iaa.imgcorruptlike.Spatter(severity=(params[0],params[1]))(image=image)
            elif(aug_id==7):
                image = iaa.BlendAlphaSimplexNoise(iaa.EdgeDetect(1))(image=image)
            elif(aug_id==8):
                image = iaa.Rotate(rotate=(params[0],params[1]))(image=image)
            elif(aug_id==9):
                image = iaa.Affine()(image=image) #to be implemented
            elif(aug_id==10):
                image = iaa.MotionBlur(k=params[0],angle=(params[1],params[2]))(image=image)
            elif(aug_id==11):
                image = iaa.imgcorruptlike.ZoomBlur(severity=(params[0],params[1]))(image=image)
            elif(aug_id==12):
                image = iaa.AddToBrightness()(image=image) #to be implemented
            elif(aug_id==13):
                image = iaa.ChangeColorTemperature(kelvin=(params[0],params[1]))(image=image)
            elif(aug_id==14):
                image = iaa.SigmoidContrast()(image=image) #to be implemented
            elif(aug_id==15):
                image = iaa.Cutout(nb_iterations=(params[0],params[1]),size=params[2],squared=params[3])(image=image)
            else:
                print("Not implemented")
        augmented.append(image)
    return augmented
    