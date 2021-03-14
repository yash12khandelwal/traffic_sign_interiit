import os
import os.path as osp
import pandas as pd
import sys
from collections import defaultdict, namedtuple
import csv

def make_dir(path):
    """ Utility function to make directories

    Args:
        path (str): Folder to make
    """

    if not osp.exists(path):
        os.makedirs(path)

def get_classes(path):
    """
    Get New class ids
    """

    return list(map(int, os.listdir(path)))

Annotation = namedtuple('Annotation', ['filename', 'label'])
def make_annotations(datapath):
    """ Making annotations for dataset class

    Args:
        filepath (str): Folder location for class

    Returns:
        list: List of Annotations
    """

    classes = get_classes(datapath)
    annotations = []
    for classid in classes:
        path = osp.join(datapath, f'{classid}')
        annotation = list(map(lambda filename: Annotation(osp.join('dataset', path, filename), classid), os.listdir(path)))
        annotations.append((classid, annotation))

    return annotations

def write_annotations(annotations, filepath):
    """
    Function that writes the annotations to file
    """

    data = dict()
    data['Filename'] = []
    data['ClassId'] = []
    for filename, label in annotations:
        # print(filename, label)
        data['Filename'].append(filename)
        data['ClassId'].append(label)
    df = pd.DataFrame(data=data)
    df.to_csv(filepath, sep=';', index=False)

def split_train_val_test_sets(path, annotations, validation_fraction, test_fraction):

    train_path = osp.join(path, 'train')
    validation_path = osp.join(path, 'valid')
    all_path = osp.join(path, 'all')
    test_path = osp.join(path, 'test')

    make_dir(train_path)
    make_dir(validation_path)
    make_dir(all_path)
    make_dir(test_path)

    test_annotation = []

    for i, annotation in annotations:
        make_dir(os.path.join(all_path, f"{i:04}"))
        make_dir(os.path.join(train_path, f"{i:04}"))
        make_dir(os.path.join(validation_path, f"{i:04}"))

        test_size = int(len(annotation) // 1 * test_fraction) * 1
        train_size = len(annotation)-test_size
        validation_size = int(train_size // 1 * validation_fraction) * 1 + test_size

        write_annotations(annotation[test_size:], os.path.join(all_path, f'{i:04}', f'GT-{i:04}.csv'))
        write_annotations(annotation[test_size:validation_size], os.path.join(validation_path, f'{i:04}', f'GT-{i:04}.csv'))
        write_annotations(annotation[validation_size:], os.path.join(train_path, f'{i:04}', f'GT-{i:04}.csv'))

        test_annotation.extend(annotation[:test_size])

    write_annotations(test_annotation, osp.join(test_path, 'GT-Test.csv'))

def prepare_train_val_n_test(source_path, validation_fraction=0.2, test_fraction=0.2):
    """ Prepare Train/Valid from raw dataset

    Args:
        validation_fraction (float, optional): valid/val split. Defaults to 0.2.
    """

    path = 'EXTRA'

    annotations = make_annotations(source_path)
    split_train_val_test_sets(path, annotations, validation_fraction, test_fraction)

if __name__ == '__main__':
    extra_classpath = sys.argv[1]

    prepare_train_val_n_test(extra_classpath, validation_fraction=0.2, test_fraction=0.2)
