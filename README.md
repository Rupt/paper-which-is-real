# Supporting code for:<br>A method to challenge symmetries in data with self-supervised learning
arXiv: https://arxiv.org/abs/2111.05442, Zenodo: https://doi.org/10.5281/zenodo.6861702 \
Test symmetries with sklearn decision tree models.

# Setup
Begin from an environment with a recent version of python 3.  \
We recommend using anaconda:
```bash
conda create -n which-is-real python==3.10.4
```
followed by
```bash
conda activate which-is-real
```
Install and activate a python virtual environment with our required packages:
```bash
source setup.sh
```
You can leave the environment with `deactivate`.  \
To clean up fully, remove the installation directory `env/`.

# Run experiments
We include scripts to reproduce all experiments from the paper and a few others.  \
To run all, execute:
```bash
make
```
This dumps plots into the various directories `example_*/`.

We describe each in a little more detail below.

## Experiment 1: Gappy detector
The cylindrical particle detector with holes and varying efficiencies.  \
In the paper, this is illustrated in Figures 2 and 3, and discussed in Section 5.

Notebook version: `example_ring_paper.ipynb`.  \
You should be able to view this on GitHub with all figures and other outputs.

Execute the plain python version with:
```bash
make example_ring_paper.log
```
This reproduces plots from the paper and dumps them into the directory `example_ring_paper/`.  \

## Experiment 2: Height map
The topographical map from Figure 4 and Section 6 of the paper.

Notebook version: `example_map_paper.ipynb`.  \
You should be able to view this on GitHub with all figures and other outputs.

Execute the plain python version with:
```bash
make example_map_paper.log
```
This reproduces plots from the paper and dumps them into the directory `example_map_paper/`.

## Bonus experiments:
* `make example_ring.log`
  * Similar to Experiment 1, plotted in 2D, showing both real and fake (transformed) data.
  * Plots dump to `example_ring/`.
* `make example_map.log`
  * Similar to Experiment 2, with more variety in its filtering symmetry-breaking wave.
  * Plots dump to `example_map/`.
* `make example_step.log`
  * A minimal example testing translation symmetry on the unit interval.
  * Uses rejection sampling for fakes.
  * Plots dump to `example_step/`.

These `make` commands run corresponding `example_*.py` scripts
after ensuring that input data are prepared.
