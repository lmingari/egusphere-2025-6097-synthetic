# Variational Autoencoder for Volcanic Ash / Synthetic experiment

## Python Dependencies

All required Python packages are listed in `requirements.txt`. Before installing them, it is good practice to create a **virtual environment**: an isolated space where these packages are installed, so they don't interfere with other Python projects on your computer.

The example below uses [`uv`](https://github.com/astral-sh/uv), a fast, modern tool for managing Python environments and packages. This is just one option, other popular alternatives include `venv` (built into Python) and `conda`. Feel free to use whichever tool you are most comfortable with; the commands below are specific to `uv`, but the same general steps (create an environment, activate it, install dependencies) apply to any of them.

To create and activate an environment with `uv`:

```bash
uv venv my_env
source my_env/bin/activate
```

Then install the dependencies into the environment:

```bash
uv pip install -r requirements.txt
```

## Training Dataset

The training, validation and test datasets used for the synthetic experiments can be obtained by running the following bash script:

```bash
./fetch_data.sh
```

## Training the Variational Autoencoder

The VAE training procedure is implemented in `VAE_training.py`. Configuration and hyperparameters are defined in separate sections of the `config.ini` file. For example, to train the network using the configuration defined in the `L16B06` section, run:

```bash
./VAE_training.py --section L16B06
```

For further information, see the help menu:

```bash
./VAE_training.py --help
```

## Training Configuration

An example configuration section from `config.ini` is shown below:

```ini
[L16B06]
LATENT_DIM = 16
LEARNING_RATE = 1E-3
NUM_EPOCHS = 400
BETA = 6
MINVAL = 0
MAXVAL = 20
```

### General Hyperparameters

* `LEARNING_RATE`: Learning rate.
* `NUM_EPOCHS`: Total number of training epochs.
* `LATENT_DIM`: Dimensionality of the latent space.

### Beta Variational Autoencoder

A $\beta$-VAE (beta variational autoencoder) modifies the standard VAE loss function by introducing a hyperparameter $\beta$ that weights the KL divergence (Kullback-Leibler divergence) term:

$$ Loss = \text{Reconstruction Loss} + \beta \times \text{KL Divergence} $$

A constant `BETA` is used here:

* `BETA`: Value of $\beta$.

### Normalization

Raw data is normalized according to the transformation:

$$ x = \frac{x_{raw} - \text{MINVAL}}{\text{MAXVAL} - \text{MINVAL}} $$

* `MINVAL`: Minimum value of the data, used as the lower bound for normalization.
* `MAXVAL`: Scale (unit) used to normalize the data. This does not necessarily correspond to the actual maximum of the data (normalized values are not clipped and may exceed 1).
