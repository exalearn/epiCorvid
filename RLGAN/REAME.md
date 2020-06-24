# Running corvid predicitions for RL training
Here there is code to train and use a neural network model to advance mock corvid simulations under various control parameters. The setup is similar to a conditional GAN, with the exception that the generator network does not sample any noise vector and instead takes the current state of the simulation (new and cumulative symptomatic cases per tract and age group) as input, then produces the prediction for the next week.

### Requirements
The code has been developed and run with the following software:
* numpy, scipy, matplotlib, h5py (python 3.7)
* pytorch 1.5.0

### Sample pre-trained weights
A model checkpoint from the `exp_adv_256` GAN is available [here](https://portal.nersc.gov/project/m3623/pretrained/RLtest/).

### Running predictions for pre-trained model
The `inference.py` script is an example script demonstrating how to use the trained model to advance the state of a simulation over some weeks. Usage is as follows:
```
python inference.py --yaml_config./config.yaml --data=/path/to/data --saved_weights=/path/to/pretrained/weights
```
The model was trained on states randomly sampled from the [RLtest](https://portal.nersc.gov/project/m3623/datasets/RLtest/) dataset, between weeks 3 and 45 (these starting/ending weeks may change in future models), so prediciton works by randomly sampling a "starting" state from the start week across the full dataset. At the start week, no controls are in place, and all simulations are coming from the same sets of parameters/controls. Then, the script will iteratively predict the following weeks of behavior, sampling a new set of control parameters each week. Currently the script is configured to make predictions for just one trajectory, but this can be easily parallelized by increasing the `bs` parameter.

