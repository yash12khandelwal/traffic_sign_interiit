import imageio
import imgaug as ia
import cv2
import imgaug.augmenters as iaa

def get_preview(images, augmentationList, probability):
    """
    Accepts a list of images and augmentationList as input.
    Provides a list of augmented images in that order as ouptut.
    """
    # augmented = []
    # for image in images:
    #     for augmentation in augmentationList:
    #         aug_id = augmentation['id']
    #         params = augmentation['params']
    #         if(aug_id==1):
    #             image = iaa.SaltAndPepper(p=params[0], per_channel=params[1])(image=image)
    #         elif(aug_id==2):
    #             image = iaa.imgcorruptlike.GaussianNoise(severity=(params[0],params[1]))(image=image)
    #         elif(aug_id==3):
    #             image = iaa.Rain(speed=(params[0],params[1]),drop_size=(params[2],params[3]))(image=image)
    #         elif(aug_id==4):
    #             image = iaa.imgcorruptlike.Fog(severity=(params[0],params[1]))(image=image)
    #         elif(aug_id==5):
    #             image = iaa.imgcorruptlike.Snow(severity=(params[0],params[1]))(image=image)
    #         elif(aug_id==6):
    #             image = iaa.imgcorruptlike.Spatter(severity=(params[0],params[1]))(image=image)
    #         elif(aug_id==7):
    #             image = iaa.BlendAlphaSimplexNoise(iaa.EdgeDetect(1))(image=image)
    #         elif(aug_id==8):
    #             image = iaa.Rotate(rotate=(params[0],params[1]))(image=image)
    #         elif(aug_id==9):
    #             image = iaa.Affine()(image=image)
    #         elif(aug_id==10):
    #             image = iaa.MotionBlur(k=params[0],angle=(params[1],params[2]))(image=image)
    #         elif(aug_id==11):
    #             image = iaa.imgcorruptlike.ZoomBlur(severity=(params[0],params[1]))(image=image)
    #         elif(aug_id==12):
    #             image = iaa.AddToBrightness(add=(params[0], params[1]))(image=image)
    #         elif(aug_id==13):
    #             image = iaa.ChangeColorTemperature(kelvin=(params[0],params[1]))(image=image)
    #         elif(aug_id==14):
    #             image = iaa.SigmoidContrast(gain=(params[0],params[1]),cutoff=(params[2],params[3]),per_channel=params[4])(image=image) #to be implemented
    #         elif(aug_id==15):
    #             image = iaa.Cutout(nb_iterations=(params[0],params[1]),size=params[2],squared=params[3])(image=image)
    #         else:
    #             print("Not implemented")
    #     augmented.append(image)
    # return augmented
    augmentations = []
    for augmentation in augmentationList:
        aug_id = augmentation['id']
        params = augmentation['params']
        if(aug_id==1):
            aug = iaa.SaltAndPepper(p=params[0], per_channel=params[1])
        elif(aug_id==2):
            aug = iaa.imgcorruptlike.GaussianNoise(severity=(params[0],params[1]))
        elif(aug_id==3):
            aug = iaa.Rain(speed=(params[0],params[1]),drop_size=(params[2],params[3]))
        elif(aug_id==4):
            aug = iaa.imgcorruptlike.Fog(severity=(params[0],params[1]))
        elif(aug_id==5):
            aug = iaa.imgcorruptlike.Snow(severity=(params[0],params[1]))
        elif(aug_id==6):
            aug = iaa.imgcorruptlike.Spatter(severity=(params[0],params[1]))
        elif(aug_id==7):
            aug = iaa.BlendAlphaSimplexNoise(iaa.EdgeDetect(1))
        elif(aug_id==8):
            aug = iaa.Rotate(rotate=(params[0],params[1]))
        elif(aug_id==9):
            aug = iaa.Affine()
        elif(aug_id==10):
            aug = iaa.MotionBlur(k=params[0],angle=(params[1],params[2]))
        elif(aug_id==11):
            aug = iaa.imgcorruptlike.ZoomBlur(severity=(params[0],params[1]))
        elif(aug_id==12):
            aug = iaa.AddToBrightness(add=(params[0], params[1]))
        elif(aug_id==13):
            aug = iaa.ChangeColorTemperature(kelvin=(params[0],params[1]))
        elif(aug_id==14):
            aug = iaa.SigmoidContrast(gain=(params[0],params[1]),cutoff=(params[2],params[3]),per_channel=params[4])
        elif(aug_id==15):
            aug = iaa.Cutout(nb_iterations=(params[0],params[1]),size=params[2],squared=params[3])
        else:
            print("Not implemented")
        augmentations.append(aug)
    
    images_augmented = iaa.Sometimes(p=probability, then_list=augmentations)(images=images)
    return images_augmented
