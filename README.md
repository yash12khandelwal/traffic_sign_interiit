# traffic_sign_interiit

Download and process the dataset using the following commands
```
cd dataset
python install_data_GTSRB.py
```

To train the model without logging on wandb
```
python train.py -v default_params
```

To train the model while logging on wandb
```
python train.py -v default_params -w
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
