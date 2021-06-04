# When in Doubt: Neural Non-Parametric Uncertainty Quantification for Epidemic Forecasting



## Setup

First install Anaconda. The dependencies are listed in `environment.yml` file. Make sure you make changes to version of `cudatoolkit` if applicable.

Then run the following commands:

```bash
conda env create --prefix ./envs/epifnp --file environment.yml
source activate ./envs/epifnp
```

## Directory structure

```
-data
	- ILINet.csv -> wILI values for seasons 2003 to 2020 collected from flusight
- model_chkp -> stores intermediate model parameters while training
- models/fnpmodels.py -> implementation of EpiFNP modules
- plots -> plots of predictions
- saves -> saves predictions for models as pkl files
- train_ili.py -> training script for EpiFNP
- test_ili.py -> inference of trained model
- test_regress.py -> Autoregressive inference using a trained model
```

## Training

Run:

```
python train_ili.py -y <test season> -w <week ahead> -a trans -n <experiment name> -e <max num. of epochs>
```

Or run `run.py` to run all experiments.

Prediction plots will be saved in `plots/Test<experiment name>.png` and model in `model_chkp` folder.

## Inference

Run:

```bash
python test_ili.py -y <test season> -w <week ahead> -a trans -n <experiment name>
```

for normal inference.

Run: 

```bash
python test_regress_ili.py -y <test season> -w <week ahead> -a trans -n <experiment name>
```

for auto-regressive inference. Note: Train and use a 1 week ahead model for AR inference.

The predictions and plots are saved in `saves` and `plots` respectively.
