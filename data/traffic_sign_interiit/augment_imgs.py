from options.train_options import TrainOptions
from augments.augs import load_augments
import os
import os.path as osp
import random
import cv2

if __name__ == '__main__':
    opts = TrainOptions()
    args = opts.initialize()

    DATA_DIR = args['experiment'].data_dir
    CLASS = args['experiment'].class_id
    NUM_SAMPLES = args['experiment'].samples
    size = tuple(args['experiment'].size)
    DATA_DIR = osp.join(DATA_DIR, 'Final_Training', 'Images', f'{CLASS}'.rjust(5, '0'))
    imgs = os.listdir(DATA_DIR)
    imgs = [ osp.join(DATA_DIR, img) for img in imgs ]

    random_samples = random.choices(imgs, k=NUM_SAMPLES)

    for i, img in enumerate(random_samples):
        if img.endswith('.csv'):
            continue
        image = cv2.imread(img)
        image = cv2.resize(image, size)
        image = load_augments(args['augmentations'], top=1)(image=image)

        cv2.imwrite(img[:-4]+'_aug'+img[-4:], image)
    print('Total Samples', len(set(random_samples)))
