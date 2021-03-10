import os
import zipfile
import shutil
from collections import defaultdict, namedtuple
import csv

def download_gtsrb():
    """ Function to download raw dataset if not downloaded before
    """

    TMP_DATA_DIR = "."
    TMP_LABELS_DIR = os.path.join(TMP_DATA_DIR, "GTSRB/Final_Test")

    training_imgs_link = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip'
    test_imgs_link = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip'
    test_gt_link = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip'

    make_dir('zips')
    os.system(f'wget -c {training_imgs_link} -P zips')
    os.system(f'wget -c {test_imgs_link} -P zips')
    os.system(f'wget -c {test_gt_link} -P zips')

    to_unpack = [
        ("zips/GTSRB_Final_Training_Images.zip", TMP_DATA_DIR),
        ("zips/GTSRB_Final_Test_Images.zip", TMP_DATA_DIR),
        ("zips/GTSRB_Final_Test_GT.zip", TMP_LABELS_DIR)
    ]
    
    for file, directory in to_unpack:
        print("Unzipping {} to {}...".format(file, directory))
        with zipfile.ZipFile(file,"r") as zip_ref:
            zip_ref.extractall(directory)

def make_dir(path):
    """ Utility function to make directories

    Args:
        path (str): Folder to make
    """

    if not os.path.exists(path):
        os.makedirs(path)

Annotation = namedtuple('Annotation', ['filename', 'label'])
def read_annotations(filename):
    """ Reading annotations from test csv file

    Args:
        filename (str): File location for test csv file

    Returns:
        list: List of Annotations
    """

    annotations = []
    
    with open(filename) as f:
        reader = csv.reader(f, delimiter=';')
        next(reader) # skip header

        # loop over all images in current annotations file
        for row in reader:
            filename = row[0] # filename is in the 0th column
            label = int(row[7]) # label is in the 7th column
            annotations.append(Annotation(filename, label))
            
    return annotations

def load_training_annotations(source_path, num_class):
    """ Reading train annotations for each class

    Args:
        source_path (str): Path of training data

    Returns:
        [type]: List of Annotations
    """

    annotations = []
    for c in range(0,43):
        filename = os.path.join(source_path, format(c, '05d'), 'GT-' + format(c, '05d') + '.csv')
        annotations.extend(read_annotations(filename))
    return annotations

def copy_files(label, filenames, source, destination, move=False):
    """ Copy files from source to destination

    Args:
        label (str): Path for labels
        filenames (list): Files that need to be copied
        source (str): Source Destination
        destination (str): Target Destination
        move (bool, optional): Move files instead of copying. Defaults to False.
    """

    func = os.rename if move else shutil.copyfile
    
    label_path = os.path.join(destination, str(label))
    if not os.path.exists(label_path):
        os.makedirs(label_path)
        
    for filename in filenames:
        destination_path = os.path.join(label_path, filename)
        if not os.path.exists(destination_path):
            func(os.path.join(source, format(label, '05d'), filename), destination_path)

def split_train_validation_sets(source_path, train_path, validation_path, all_path, num_class, validation_fraction=0.2):
    """ Spliting the Train folder into train and valid

    Args:
        source_path (str): Source destination of Train
        train_path (str): Final path of the train set
        validation_path (str): Final path of the valid set
        all_path (str): Final path of the total (train+valid) set
        validation_fraction (float, optional): Split fraction . Defaults to 0.2.
    """
    
    make_dir(train_path)
        
    make_dir(validation_path)
        
    make_dir(all_path)
    
    annotations = load_training_annotations(source_path, num_class)
    filenames = defaultdict(list)
    for annotation in annotations:
        filenames[annotation.label].append(annotation.filename)

    for label, filenames in filenames.items():
        filenames = sorted(filenames)
        
        validation_size = int(len(filenames) // 30 * validation_fraction) * 30
        train_filenames = filenames[validation_size:]
        validation_filenames = filenames[:validation_size]
        
        copy_files(label, filenames, source_path, all_path, move=False)
        copy_files(label, train_filenames, source_path, train_path, move=True)
        copy_files(label, validation_filenames, source_path, validation_path, move=True)

def prepare_test():
    """ Function to prepare Test set from raw dataset
    """

    make_dir('GTSRB/test')

    os.system('mv GTSRB/Final_Test/Images/*.ppm GTSRB/test')
    os.system('mv GTSRB/Final_Test/GT-final_test.csv GTSRB/test/.')

def prepare_train_n_val(validation_fraction=0.2, num_class=43):
    """ Prepare Train/Valid from raw dataset

    Args:
        validation_fraction (float, optional): valid/val split. Defaults to 0.2.
    """

    path = 'GTSRB'
    source_path = os.path.join(path, 'Final_Training/Images')
    train_path = os.path.join(path, 'train')
    validation_path = os.path.join(path, 'valid')
    all_path = os.path.join(path, 'all')
    split_train_validation_sets(source_path, train_path, validation_path, all_path, num_class, validation_fraction)


if __name__ == "__main__":
    num_class = 43  # Change this if classes change
    download_gtsrb()
    prepare_test()
    prepare_train_n_val(validation_fraction=0.2, num_class=num_class)