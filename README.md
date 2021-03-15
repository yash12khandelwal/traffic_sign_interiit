# traffic_sign_interiit

Download and process the dataset using the following commands
```
# This will download and create annotations for GTSRB dataset
cd dataset
python install_data_GTSRB.py
```

Adding new classes 
```
cd dataset
python prepare_new_classes.py [new_class_path] [write_path]

Note:
Format for new_class_path
new_class_path
|___ 43
|___ 44
|___ 45
    ...
All images of class_id 43 must be in folder 43, class_id must be integers
```

To train the model
```
python train.py -v [config_file_name] [-w|--wandb]
Use --wandb option to enable logging in wandb
```

Default Config Files

- Augmentations
  - default_augment_conf.json
- Model Parameters
  - default_params.json

Baselines

- MicroNet
```
Validation Accuracy = 98.57
Test Accuracy = 98.59
```

- DKS
```
Validation Accuracy = 96.475
Test Accuracy = 96.53
```
