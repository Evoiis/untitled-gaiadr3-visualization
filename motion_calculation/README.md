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