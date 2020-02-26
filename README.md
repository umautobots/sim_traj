# Stochastic Sampling Simulation for Pedestrian Trajectory Prediction
by Cyrus Anderson and Xiaoxiao Du at [UM FCAV](https://fcav.engin.umich.edu/)

### Introduction
This paper presents a method to simulate realistic synthetic pedestrian trajectories based on
small amounts of real annotated data. This work was presented at IROS 2019 in Macau, China.
For more details, please refer to our published paper in
IEEE Xplore (https://ieeexplore.ieee.org/document/8967857)
or the arxiv version (https://arxiv.org/abs/1903.01860).

### Citation
If you find this paper helpful, please consider citing:
```
@inproceedings{anderson2019stochastic,
  title={Stochastic Sampling Simulation for Pedestrian Trajectory Prediction},
  author={Anderson, Cyrus and Du, Xiaoxiao and Vasudevan, Ram and Johnson-Roberson, Matthew},
  booktitle={2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={4236--4243},
  year={2019},
  organization={IEEE}
}
```

## Steps

1) First we simulate trajectories to create the synthetic data.
```
python sampler.py
```

2) Next we can save these trajectories into formatted training data for training a neural network.
This will create a `training` folder if it doesn't already exist, and a folder with the training data.
```
python sim2training.py
```
The file structure wil look like:
```
datasets/
training/
|__synth_large/
    |__split_1.0_0/
         |__eth/
```
with the `eth` folder containing synthetic training data from all `synth_large` datasets except ETH.

3) Train the neural network of our choice on the data (such as [Social GAN](https://github.com/agrimgupta92/sgan)).

For a model trained on the `eth` folder data, we can save it at:
```
datasets/
training/
models/
|__synth_large/
    |__split_1.0_0/
         |__eth.pt
```
so this will be the model to make predictions on ETH.


4) Evaluate the trained model on each scene.

This expects a function to load the trained model so we can make predictions, one of:
- a function that returns a `Predictor` object
- `make_pred_fcn`: convenience function making `Predictor` objects for Social GAN-style models

Once one of these has been specified, we can run
```
python evaluate_split.py
```



### Dependencies

- numpy
- PyTables
- pandas
- tqdm
