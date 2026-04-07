# Orbit Integration

Planned to use galpy orbit integrate directly for motion.

Also attempting to train an MLP(multi-layer perceptron) to create a speedier alternative.


## Dependencies

Cuda 13.0.0 Install
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda-repo-ubuntu2404-13-0-local_13.0.0-580.65.06-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-13-0-local_13.0.0-580.65.06-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-13-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-0
```


### Tmp Notes
Trained first orbit_mlp.pt model on training_data.
1000000 stars. 100 epochs.

~75 parsecs of error.
One shot predict, cost 0.5 sec. (Should also test batch predict.)

Far from the perfect result but still lots of things to try.
- Larger Dataset
    - Marginal improvement
        - Doesn't seem like lack of data is a major issue at the moment
- SiLU over relu
    - Significant improvement with silu
- Changing input features
    - Time --> Fourier features mapping
    - add r0 = sqrt(x0,y0)
- Edit hidden layers


Iterations
#### 1:
- 1 Million stars * 100 timesteps, 7 features
~ 75 parsecs err

#### 2:
- 2 Million stars * 100 timesteps, 7 features
~ 60 parsecs err

#### 3:
- Swapped from relu to silu
~ 21 parsecs error

#### 4:
- Added r0 and time fourier features
- Trained on 1/2 Training data (Had issues with memory)
~59 parsecs error

#### 5:
- Trained on 3/4 of training_data + training_data_2
~ 40 parsecs error

Stopped early because it started plateauing at 40.
(also forgot to swap the learning rate when swapping data sets)
    - no longer a problem because I added load_checkpoint

#### 6: (training_data_3, orbit_norm_6)
- training_data_3, Reduced Input Time space from -3,3 to -1,1
- Epochs, increased from 100 to 150
- scheduler changes, min_lr = 1e-6, patience=15
~16 parsecs error, after first half of data
~10 parsecs error, after second half of data
Time taken(model_test.py), 1.3s

Still room to improve here based on remaining learning rate.

#### 6 continued:
HIDDEN=[256, 256, 256, 128] (No change from before)
- Using mixed precision brought epoch time from ~11 to ~8 seconds
~10 parsecs, after first half
    - maybe patience is too high?
~10 parsecs, after second half


#### 7: (training_data_3, orbit_norm_6)
- Changed hidden layers, From [256, 256, 256, 128] to [512, 512, 256, 256, 128]
- patience=15
- Significantly slower by ~2x, to train and to test
~10 parsecs error, after first half of data
~ 5.6 parsecs error, after second half of data
Time taken(model_test.py), 2.15


#### 8: (training_data_3, orbit_norm_6)
- Optimizer Patience reduced to 10
- HIDDEN updated to [256] * 4
- Switch to **Residual MLP**

~23 parsecs error, after first half of data
~16 parsecs error, after second half of data


#### 9: (training_data_3, orbit_norm_6)
- Back to **Plain MLP**
- Hidden = [1024, 1024, 512, 256]
~20 after first half of data
~15 parsecs error, after second half of data


#### 10: (training_data_3, orbit_norm_6)
- Repeat 9, with patience = 15
    - Patience sanity check
~30 parsecs error after first half of data


#### 11: (training_data_3, orbit_norm_6)
- Hidden = [1024, 1024, 512, 256, 128]
- Patience = 10
~26 parsecs error, after first half of data

- large jumps in loss when learning rate halves
```
Epoch 92/150  train_loss=3.10e-05  val_loss=2.88e-05  lr=5.00e-04  train_pc=np.float64(63.255904091693274), val_pc=np.float64(60.959115343239446)  time=32.0s
Epoch 93/150  train_loss=3.50e-05  val_loss=1.52e-04  lr=5.00e-04  train_pc=np.float64(67.16801653764989), val_pc=np.float64(140.0430933581815)  time=31.4s
Epoch 94/150  train_loss=2.85e-05  val_loss=3.58e-05  lr=5.00e-04  train_pc=np.float64(60.56533757900817), val_pc=np.float64(67.97838762037871)  time=32.2s
Epoch 95/150  train_loss=3.10e-05  val_loss=3.08e-05  lr=5.00e-04  train_pc=np.float64(63.16531131179766), val_pc=np.float64(62.97363789392714)  time=31.7s
Epoch 96/150  train_loss=3.20e-05  val_loss=9.16e-05  lr=2.50e-04  train_pc=np.float64(64.24828934771149), val_pc=np.float64(108.641003260303)  time=31.6s
Epoch 97/150  train_loss=1.21e-05  val_loss=1.03e-05  lr=2.50e-04  train_pc=np.float64(39.48829583949843), val_pc=np.float64(36.49230998602964)  time=31.3s
Epoch 98/150  train_loss=1.01e-05  val_loss=1.01e-05  lr=2.50e-04  train_pc=np.float64(36.11891212004395), val_pc=np.float64(36.020122083804154)  time=31.6s
Epoch 99/150  train_loss=9.87e-06  val_loss=9.85e-06  lr=2.50e-04  train_pc=np.float64(35.67020919499895), val_pc=np.float64(35.627068512536944)  time=31.3s
```

#### 12: (training_data_3, orbit_norm_6)
- Add gradient clipping to manage loss instability
- Swap to cosine annealing scheduler
- Epochs = 200

~21 parsecs error, after first half
- issue with scheduler learning rate on continue
    - need to tune first and second half to different mins

~30seconds per epoch

#### 13: (training_data_3, orbit_norm_6)
- [512, 512, 256, 256, 128] (from iteration 7)
- Back to plateau scheduler
- Patience = 20
- Epochs = 200

~11seconds per epoch
~17 parsecs error, after first half
~13 parsecs error, after second half (lr=3.13e-05)

3rd pass
- patience = 15
- 90 epochs on first half of data
~12 parsecs error

---


#### 14: todo
- Same as 13 but with cosine annealing scheduler



#### x: todo (training_data_4, orbit_norm_13)
- Generate new training set
    - Add larger test and validation sets (200000 stars instead of 100000)

