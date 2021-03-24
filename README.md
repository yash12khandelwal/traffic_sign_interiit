# traffic_sign_interiit

## Setup

### Steps:

1. Install requirements
    ```
    pip install -r requirements.txt
    ```

2. Setup GTSRB dataset: This is will download the dataset, split the training set into training and validation sets, and create annotations for the same

    ```
    cd dataset
    python install_data_GTSRB.py
    ```

    The GTSRB dataset will be setup in the following format:

    ```
    dataset
    ├── GTSRB
        ├── Final_Training      # Downloaded raw training dataset
        ├── Final_Test          # Downloaded raw test dataset
        ├── train               # Contains subdirectories for all classes with annotations stored in a CSV file
        ├── val                 # Same as train
        ├── test                # Contains a CSV file with test annotations
        ├── all                 # Training and Validation set combined
        ...                  
    ├── zips                    # Zip files containing raw train and test dataset   
    ...
    ```

3. Setup 5 new classes: This will split the dataset into train, val and test set. Annotations will be made for the same and it will save them to ```extra_path```
    ```
    cd dataset
    python prepare_new_classes.py [new_class_path] [extra_path]
    ```

    Note: Format for new_class_path
    ```
    new_class_path
    ├── 43
    ├── 44
    ├── 45
    ...
    ```
    Note: All images of class_id 43 must be in folder 43, class_id must be integers

    This will setup the extra classes in a similar format as the GTSRB dataset:
    ```
    extra_path
    ├── train               # Contains subdirectories for all classes with annotations stored in a CSV file
    ├── val                 # Same as train
    ├── test                # Contains a CSV file with test annotations
    └── all                 # Contains all the annotations for all images in new_class_path
    ```

4. (Optional) Class weights can be created to combat class imbalance in the dataset
    ```
    cd dataset
    python create_class_weights.py --data-dir [GTSRB_dataset_directory] --extra-dir [extra_classes_write_path] --weights [average/log]
    ```
    This will create an .npy file in the config directory which can be specified in [config/params.json](./config/params.json)

## Default Config File

This file contains the default parameters on which all experiments were conducted

- [config/params.json](./config/params.json)

The structure of the config file is as follows. The experiment part contains the hyperparameters used (learning rate, weight decay, class weights etc.) and the model chosen. Here, you will also have to specify the directory where the annotations for extra classes are stored as well as the GTSRB dataset directory. The augmentations dictionary stores which augmentations are to be applied and with what parameters. Lastly, a probability of whether the augmentation is applied or not is specified.

    {
        "experiment": {
            ...
            ...
        },
        "augmentations": {
            "Use": {
                ...
            },
            ...
            ...,
            "probability": ...
        }
    }

## Training the model

```
python train.py -v [config_file_name] [-w|--wandb]
```

Use --wandb option to enable logging in wandb

## Baselines

- ### MicronNet
```
Validation Accuracy = 98.57
Test Accuracy = 98.59
```
