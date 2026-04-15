
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
**Stopped Early**

#### 5:
- Trained on 3/4 of training_data + training_data_2
~ 40 parsecs error
**Stopped Early**

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

(test_data_3) 5.767164 parsecs error
(test_data_12) 7.69645 parsecs error


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


#### 12: (training_data_3, orbit_norm_6)
- Add gradient clipping to manage loss instability
- Swap to cosine annealing scheduler
- Epochs = 200

~21 parsecs error, after first half
- issue with scheduler learning rate on continue
    - need to tune first and second half to different mins
    - or add override

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


#### 7.1:
- Attempt to replicate iteration 7
- Copied all settings from iteration 7 but with current code.
    - Epoch = 150, patience = 15, H = [512, 512, 256, 256, 128]

~28 parsecs error, after first half
~19 parsecs error, after second half

- Definitely feels like I tweaked something wrong in the iterations between now and 7
- Need to build config input and log to file before continuing

#### 7.2:
- 7.1 but with cosannealing scheduler
~27 parsecs error, after first half
~20 parsecs error, after second half


#### 7.3:
- No dataset split
~11 parsecs error

#### 7.4:
- AdamW
~ 12 parsecs error

#### 7.5:
- No gradient clipping
~ 10.5 parsecs error

#### 7.6:
- 7.5 but cosanneal
~10 parsecs error

#### 7.7:
- data split and learning rate reset
~18

#### 7.8:
- 450 epochs
~ 10 parsecs error

### 7.9:
- cosanneal warm restart, t0: 50, mult: 2
- 350 Epochs
~10.66 parsecs error

#### 14:
- [1024, 512, 512, 256, 256, 128]
~8.1 parsecs error

#### 15: (training_data_4)
- [512, 512, 256, 256, 128]
~13 parsecs error

#### 16
- Residual model
- Very slow and very unstable

#### 17: (Training_data_5) (close far uniform dist_kpc dataset)
~10.9 (test_data_5) parsecs
~9.6 (test_data_3) parsecs

#### 20: (Training_data_12) (flat dist_kpc sampling)
- 150 epochs
~15.698435437964946 parsecs error

- 300 epochs
~10.16 parsecs error
~7.5 (test_data_3) parsecs error

- 140 more epochs cosannealhuber
- 150 more epochs huber plateau, reset starting lr to 0.0001
~5.9 parsecs error

(test_data_3) 7.51196 parsecs error
(test_data_12) 8.395 parsecs error


#### 21: (Training data 12)
- Added Layernorms in after each hidden layer, before silu
- Using huber loss_fn with delta 0.1
- H-layer change: [1024, 1024, 512, 512, 256]
- batch size reduced to 16384
- swapped from fp16 to bfloat16

(test_data_3) ~15.5 parsecs error

#### 22: (Training data 12)
- layernorm off, hlayers back to 512,512,256,256,128
- mse loss
- batch size reduced to 16384
- using bfloat16

- After 300 epochs:
(test_data_3) ~12.8 parsecs error

#### 23:
- 22 with batch size reduced to 8192
- 150 epochs: ~15.507936918409243 parsecs error
- 300 epochs: 14.924628728195806 parsecs error
- using bfloat16

#### 24:
- 23 with batch size 102400
- 150 epochs: ~27.889 parsecs error
- 300 epochs: 17.07351964574024 parsecs error
(test_data_3) 15.86816 parsecs error 
- using bfloat16

#### 25:
- SIREN, first run
- using bfloat16

#### 26:
- SIREN
- 150 epochs in swap to huber and override learning rate to 0.0001
- huber delta: 0.1
- using bfloat16

68.663 parsecs error

#### 27:
- 26, but huber delta: 0.1
- using bfloat16

64.146 parsecs loss


#### 27.1:
- 16384 batch size
- start_learning rate 0.005 from epoch 0
- using bfloat16

4220 parsecs error
popped at the start due to high learning rate

#### 27.2:
- hidden layer shrink [512, 256, 256, 128]
- using bfloat16

86 parsecs error

#### 27.3:
- omega = 60
- using bfloat16

103 parsecs error

#### 27.4:
- omega = 10
- using bfloat16


#### 30:
- Plain MLP, similar to iteration 20
- using bfloat16

- After 150 epochs:
    - 22.869 parsecs test error

- After 300 epochs:
    - 17.2679 parsecs test error

- After 450 epochs:
    - 16.794 parsecs test error:

- After 600 epochs:
    - 16.3289 parsecs test error:

#### 30.1:
- using fp16

- After 150 epochs: 
    - 16.8988 parsecs test error

- After 300 epochs:
    - 13.0536 parsecs test error

- After 450 epochs:
    - 11.471961 parsecs test error

- After 600 epochs:
    - 10.975 parsecs test error

- Looks like bf16 is having a detrimental effect on training...

#### 31: (training_data_13S)
- 13S data, 100k stars instead of 1M
- fp16
- 65.459 parsecs test error

- second run, no changes
- 54.796 parsecs test error

- third run, cast to float
- 55.791 parsecs test error

#### 31.1: (training_data_13S)
- 13S data, 100k stars instead of 1M
- fp32
- 342.569 parsecs test error

- second run, with gradscaler and grad clipping
- 46.023 parsecs test error

- third run, no autocast, no gradscaler, with grad clipping
- 48.7197

- **Looks like autocast is negatively affecting fp32**

- fourth run, no autocast, no gradscaler, with grad clipping
- 46.684 parsecs test error

- 5th run, no changes, with grad clipping
- 47.102 parsecs test error

#### 31.1.1:
- 31.1 but with no grad clipping
- 

#### 31.2: (training_data_13S)
- 13S data, 100k stars instead of 1M
- bf16
- 66.330 parsecs test error

- second run, cast to float, no autocast
- 64.188 parsecs test error

#### 32:
- 31, fp32, 600 epochs

#### 32.1:
- 31.1, 600 epochs, with grad clipping

#### 32.2:
- 31.2, 600 epochs
- 
