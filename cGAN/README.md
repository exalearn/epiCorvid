# Generating corvid samples with a conditional GAN

Here there is code to train and use a GAN to generate mock runs of corvid for different parameters, using conditional batch normalization to condition the GAN on parameter values.

### Requirements
The code has been developed and run with the following software:
* numpy, scipy, matplotlib, h5py (python 3.7)
* pytorch 1.4.0

### Sample pre-trained weights
A model checkpoint from the `3parA_20k` GAN is available [here](https://portal.nersc.gov/project/m3623/pretrained/3parA_20k/).

### Generating from pre-trained model
The `gen_pretrained.py` script is an example script that generates sample runs for the `3parA_20k` dataset, which varies the corvid parameters `R0`, `workfromhome`, and `workfromhomedays`. These parameters are linearly scaled to [-1,1] according to [these ranges](https://github.com/exalearn/epiCorvid/blob/b36765e80f321860068f60dbe40f3af17b59c34f/corvid_march/HISTORY#L149) before being fed to the GAN. Usage is as follows:
```
python gen_pretrained.py ./config/dcgan.yaml 3parA_20k checkpt_path output_path
```
Here, `checkpt_path` and `output_path` are the paths to the saved model and desired output file, respectively. User can manually adjust the number of simulations generated as well as the range of parameters used for generation by editing the script.
