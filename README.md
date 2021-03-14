# traffic_sign_interiit

Download and process the dataset using the following commands
```
cd dataset
python install_data_GTSRB.py
```

To train the model
```
python train.py -v default_params.json
```

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