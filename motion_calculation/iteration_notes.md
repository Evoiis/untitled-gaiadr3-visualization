
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

Iteration 31. Use 100k star training datset.
---
- Switching to a more focused/purposeful approach.

- Main purpose:
    - Look at training outcomes based on which datatype I use to load training data to the gpu
    - Look at training outcomes from a 100k star dataset

Results:
- Fixed fp32 autoscale issue.
- FP32 > FP16 > BF16
    - BF16 performing worse, could be because of loss of precision
        - BF16 uses more bits for the exponent
            - FP16: 5 bit exponent, 10 bit mantissa
            - BF16: 8 bit exponent, 7 bit mantissa

#### 31: (training_data_13S)
- 13S data, 100k stars instead of 1M
- fp16
- 65.459 parsecs test error

- second run, no changes
- 54.796 parsecs test error

- third run, cast to float
- 55.791 parsecs test error
- (test_data_12) 55.159 parsecs test error 

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
- (test_data_12) 46.287 parsecs test error

#### 31.1.1:
- 31.1 but with no grad clipping
- 49.821 parsecs test error

#### 31.2: (training_data_13S)
- 13S data, 100k stars instead of 1M
- bf16
- 66.330 parsecs test error

- second run, cast to float, no autocast
- 64.188 parsecs test error
- (test_data_12) 63.505 parsecs test error


Iteration 32. 600 epoch versions of Iteration 31
---
- Main purpose:
    - See how much more I can extract with the 100k star dataset by extending epochs to 600

Results:
- Pretty clear there was a bit more performance to extract.
- 32.2: lr=3.13e-05, still LR left, might not have pushed all the way to the limit
#### 32:
- 31, fp32, 600 epochs

- 42.081 parsecs test error

#### 32.1:
- 31.1, 600 epochs, with grad clipping

- 33.015 parsecs test error

#### 32.2:
- 31.2, 600 epochs

- 66.75 parsecs test error
- Last epoch: lr=1.00e-06  train_pc=np.float64(48.33429146927758), val_pc=np.float64(66.79701231983351)
    - train_pc is significantly less than val_pc


Iteration 33. Training with 200k stars dataset
---
- Each 33.X is 33.0 with one change.
- Continue with FP32.

- Main Purpose:
    - Explore variations with 200k star datset

Results:
- got to the ~7 parsecs error mark with batch size 8192
- batch size is having a much larger affect with the smaller dataset
- 600 epochs is overkill
    - 33.1 stopped improving around 380 epochs
- for obvious reasons, ramping up the dataset from small to large is much better
    - 1 million stars feels like overkill now
    - However, it was good to learn about fp16/bf16, autocasting and a bit of memory management.

- grad clipping helping a bit
- Larger model (33.4) did a bit better
- Doesn't really seem like much point to shrink hidden layers
#### 33.0: (training_data_14_20p)
- 200k stars in training data
- fp32
- 600 epochs
- batch_size 102400
- no grad clip

- lr=3.13e-05 
- 27.755 parsecs test error

#### 33.1:
- batch_size 8192

- 7.43605633162477 parsecs test error
- Error decrease went flat around 380 epochs
    - Last 2 LR halves didn't really change much

#### 33.2: (training_data_14_20p)
- batch_size 102400
- cosanneal

- 20.0712 parsecs test error

#### 33.3:
- with grad clip

- 21.572680506625503 parsecs test error
#### 33.4:
- hidden_layers: [1024, 1024, 512, 256, 128]

- 15.98415529168212 parsecs test error
#### 33.5:
- hidden_layers: [512, 512, 256, 256, 128, 64]

- 23.75804321840307 parsecs test error
### 33.6:
- hidden_layers: [512, 256, 128]

- 92.6142620665024 parsecs test error


Iteration 34. Hidden layers and datasets.
---
- 34.X Configs won't necessary be related to each other in this iteration
- Reduce epochs to 400

Main Purpose:
- Build on results from 32/33 and explore cases to improve performance
    - Compare results with 100k and 300k star datasets
    - Explore hidden layer layouts

Notes:
- Doing a cosanneal run with batch size 4096
- 

Results:
- New record of 5.21628 parsecs
- Batch size of 4096 = slower epochs but loss drops faster
    - Can reduce plateau scheduler patience and epochs with 4096
        - Or use cosanneal or multistep scheduler
- 

#### 34.0: (training_data_13S)
- Based on 33.1, 
- Use 100k star dataset
- Batch size 8192

- 10.267 parsecs test error

#### 34.1: (training_data_14_20p)
- Based on 33.1
- Batch size 4096

- 5.21628 parsecs test error 
    - NEW RECORD!!!
        - Best to run it through model_test.py next
- End LR = 7.81e-6, might be a little bit more to squeeze out here
    - Lower patience in scheduler coudl help

#### 34.2: (training_data_14_20p)
- 34.1, with Grad Clip ON
- Batch size 4096

#### 34.3: (training_data_14_20p)
- Hidden layers: [256, 256, 256, 256, 256, 256, 256, 256, 256]
    - Wider but similar amount of parameters as my current default: [512, 512, 256, 256, 128]
- Batch size 4096

#### 34.4: (training_data_14_20p)
- [1024, 512]
    - Taller but similar amount of parameters as my current default: [512, 512, 256, 256, 128]
- Batch size 4096

#### 34.5: (training_data_14_20p)
- [1024, 1024]
    - Tall!
- Batch size 4096

#### 34.6: (training_data_14_20p)
- [128, 256, 512, 512, 256, 128]
    - Grow then Shrink
- Batch size 4096

#### 34.7: (training_data_15_300k)
- Training with 300k star dataset
    - I swear this is the dataset naming scheme I'll stick with
- Default hidden layer
- Batch size 8192

Iteration 35. MLflow Integration
---
https://mlflow.org/docs/latest/ml/deep-learning/pytorch/index.html
- Integrate mlflow to track metrics during training



Future Iterations:
- Residual NN
- Without time fourier features
- Explore different activation functions
    - GELU, snake

