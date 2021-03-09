# traffic_sign_interiit

Download and process the dataset using the following commands
```
cd dataset
python install_data_GTSRB.py
```

To train the model
```
python train.py --model micronet --size 48 48 --batch-size 50 --epochs 85 --learning-rate 0.007 --weight-decay 0.00001  --device cuda
```

Baselines

- MicroNet
```
Validation Accuracy = 98.57
Test Accuracy = 98.39
```

- DKS
```
Validation Accuracy = 97.52
Test Accuracy = 97.44
```