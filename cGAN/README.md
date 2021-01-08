# Conditional GANs as surrogate models for epidemiological simulations
  
<img src="https://portal.nersc.gov/project/m3623/slides/corvidGAN/covid_header.png" alt="Corvid GAN header image" width="600"/>
  
This page documents efforts to develop a GAN-based surrogate model for simulations of a SaRS-CoV-2 epidemic in metropolitan Seattle, using the [corvid](https://github.com/dlchao/corvid) code as a baseline. For details on the epidemiological model of corvid, population parametrization, and geographic structure, see the [associated paper](https://www.medrxiv.org/content/10.1101/2020.04.08.20058487v1) written by the developers. The base corvid version for this work was cloned in March 2020, and can be found [in the parent repository of this page](https://github.com/exalearn/epiCorvid/tree/master/corvid_march).

In this repository there is code to train and use a condtional GAN to generate mock runs of corvid for different settings of epidemiological parameters as well as non-pharmaceutical interventions (NPIs). Potential use cases for such a surrogate model, which is considerably faster than running a full simulation, include rapid parameter searches, uncertainty quantification, and possibly the training of a reinforcement learning (RL) agent for learning optimal control policies to manage the pandemic.

The corpus used to build our training and validation datasets consists of 10k-20k statistically independent runs of corvid, varying both epidemiology parameters such as R0 as well as the strength of various NPI mitigation measures. With this data, we have trained multiple models to reproduce the behavior of the simulations on the aggregate level, over age groups and census tracts, up to reasonable uncertainties given the inherent stochasticity and variability seen in corvid runs. That said, **these models are NOT intended for making medical or political decisions of any kind** -- their posting here is merely for the purpose of documentation and for assisting future studies in surrogate modeling for epidemiology.

---


### Summary

Below we present the datasets, model, and results. These are also summarized in [PDF Slides](https://portal.nersc.gov/project/m3623/slides/corvidGAN_summary_slides.pdf).


#### Datasets Overview
The corvid code simulates a population of ~500k individuals in metropolitan Seattle, using census data on residency and workplaces to encode the distribution of ages, home residences, and work locations of each individual. To reduce the dimensionality of the problem, we address the surrogate modeling problem by tracking cases on the level of census tracts for each age group (0-4, 5-18, 19-29, 30-65, and 65+). For each simulation, which we run for a full year, we track the epidemic by counting the number of new symptomatic cases per day, per age group, per census tract. With 124 tracts, 5 age groups, and 365 days, we organize these counts into a 2D image of shape (124, 365, 5), where each of the 5 age bins corresponds to a different "channel". This allows us to use standard vision models/CNNs to process the simulation data efficiently. A sample of a simulation run, showing the 5 age group channels separately, is displayed below.

<img src="https://portal.nersc.gov/project/m3623/slides/corvidGAN/sample_run.png" alt="sample corvid run" width="500"/>

While the corvid code has a wealth of disease and control parameters to be varied, we focus on those which have the most significant impact on the epidemic behavior. The primary parameter controlling how effective the disease is at spreading is `R0`, the basic reproductive rate. This parameter must be empirically estimated, and relies on various underlying asumptions, but we deem those details beyond the scope of this work as we are just intending to have the surrogate grasp the more abstract concept of how rapid the pandemic progresses through the population. For NPIs, we focus on scenarios with a lockdown of variable length which involves some degree of work from home orders combined with closure of all schools. We create two main datasets, corresponding to different scenarios, for the GAN to be trained with:

* **[corvid2par_10k](https://portal.nersc.gov/project/m3623/datasets/corvid2par_10k/)** simulates a 60-day lockdown, closing all schools and having 70% of individuals work from home, which starts on a randomly sampled day during the year. This ensures even coverage of different outcomes that can occur if the lockdown is enforced at the proper time, soon after pandemic onset, or at completely useless times, such as long after the epidemic peak. Here, `R0` is varied uniformly between [0,5] to capture different speeds of disease spread. For more information, see the [dataset documentation](https://portal.nersc.gov/project/m3623/datasets/corvid2par_10k/corvid2par_10k_dataset.pdf).
* **[corvid3par_20k](https://portal.nersc.gov/project/m3623/datasets/corvid3parA_20k/)** simulates a more realistic range of `R0`, varied uniformly between 2.0 and 3.0. In this set of simulations, the school closures last 90 days, while work from home compliance is varied between 0 and 90%, and lasts between 45 and 90 days. These control measures all start on day 60 of the simulation. More details in the [dataset documentation](https://portal.nersc.gov/project/m3623/datasets/corvid3parA_20k/corvid3parA_20k.pdf)


#### Model & Results
Based on the image-like format we have organized the simulations into, we use a basic 4-layer DCGAN as a template for our surrogate model. To have control parameters affect the behavior of the generator and discriminator at each layer, we replace batch normalization with conditional instance normalization. The primary metric we choose to assess quality of GAN outputs is the "epidemic curve", tracking the total number of new symptomatic cases per day, across all age groups and census tracts. Sample output from the GAN is shown below, compared to a corvid simulation with identical control parameters for a few different samples in the corvid2par_10k dataset.

<img src="https://portal.nersc.gov/project/m3623/slides/corvidGAN/GAN_sample.png" alt="sample GAN output" width="600"/>

The general trends in each scenario are captured well, with the epidemic responding to each lockdown accordingly. In corvid, the epidemic curves for each configuration are largely fixed in shape, but peak earlier or later due to stochastic effects, so we shift the peaks to align with the reference simulation for ease of comparison. More challenging cases are when `R0` is less than 1 or 2, in which case the epidemic progresses weakly and with a lot of variability (sometimes widespread infection occurs, sometimes not, depending on who is infected and when). When there are only a handful of cases throughout the whole year, the GAN just tends to output a completely zeroed-out array (while not entirely unrealistic, this is still not ideal behavior).

Important metrics to judge quality are the mean value at the epidemic peak and the mean day of the peak, as well as the associated variance of these metrics. To track how these respond to the varied parameters in the datasets, we bin these according to the controls enforced and visualize their response across the phase space below (the 3parA_20k results are shown here for brevity; results for other datasets can be found in the slides):

<img src="https://portal.nersc.gov/project/m3623/slides/corvidGAN/true_phase_space.png" alt="phase sapce response, corvid sims" width="700"/>

The same metrics for the GAN output are shown below:

<img src="https://portal.nersc.gov/project/m3623/slides/corvidGAN/GAN_phase_space.png" alt="phase sapce response, GAN" width="700"/>

In general, the GAN predictions are in good agreement with the simulations across much of the phase space. For these particular metrics, the mean GAN prediction is within 1-2 sigma of the true simulations for much of the phase space, although with only 20k samples in the dataset, the statistics are rather sparse. The regions most troublesome are those with a lot of intrinsic variability or noise, and the GAN seems to have come up with a more smooth interpolatin over the features in the phase space. These are problems which could be helped with additional training examples, though this would make the method rather data-hungry.

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

### Contact

Peter Harrington, Lawrence Berkeley National Laboratory. 

pharrington@lbl.gov
