#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import argparse
import xarray as xr
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary

from modules.dataset import EnsembleDataset, MinMaxScale
from modules.model   import VariationalAutoencoder, VAELoss
from modules.config  import CustomConfigParser, print_config

## Training function
def train_epoch(model, loader, criterion, optimizer):
    # Set training mode
    model.train()

    total_loss  = 0.0
    total_recon = 0.0
    total_kl    = 0.0
    num_batches = 0

    for batch in loader:
        # Model prediction
        prediction, mu, logvar = model(batch)
        # Compute loss
        loss, recon, kl = criterion(prediction,batch,mu,logvar)

        # Update weight
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        total_loss  += loss.item()
        total_recon += recon.item()
        total_kl    += kl.item()
        num_batches += 1
    return {'train_loss': total_loss/num_batches, 
            'train_recon': total_recon/num_batches,
            'train_kl': total_kl/num_batches}

## Evaluation function
def evaluate_epoch(model, loader, criterion):
    # Set inference mode
    model.eval()

    total_loss  = 0.0
    total_recon = 0.0
    total_kl    = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            # Model prediction
            prediction, mu, logvar = model(batch)
            # Compute loss
            loss, recon, kl = criterion(prediction,batch,mu,logvar)

            # Update metrics
            total_loss  += loss.item()
            total_recon += recon.item()
            total_kl    += kl.item()
            num_batches += 1
    return {'validation_loss': total_loss/num_batches, 
            'validation_recon': total_recon/num_batches,
            'validation_kl': total_kl/num_batches}

def load_config(config_path="config.ini", section=None):
    # Baseline defaults
    default_config = {
        'BETA':          1.0,
        'BATCH_SIZE':    16,
        'LATENT_DIM':    512,
        'LEARNING_RATE': 1E-4,
        'NUM_EPOCHS':    150,
        'MINVAL':        0,
        'MAXVAL':        8,
        'VARKEY':        'tephra_col_mass',
        'UPSAMPLING':    'transpose',
    }

    # Initialize parser with hardcoded defaults
    parser = CustomConfigParser(defaults={k: str(v) for k, v in default_config.items()})
    parser.read(config_path)

    # Determine target section; fallback to 'DEFAULT' if None or invalid section
    target_section = section if (section and parser.has_section(section)) else 'DEFAULT'

    # Extract values using native typed getters
    return {
        'BETA':          parser.getfloat(target_section, 'BETA'),
        'LEARNING_RATE': parser.getfloat(target_section, 'LEARNING_RATE'),
        'MINVAL':        parser.getfloat(target_section, 'MINVAL'),
        'MAXVAL':        parser.getfloat(target_section, 'MAXVAL'),
        'BATCH_SIZE':    parser.getint(target_section, 'BATCH_SIZE'),
        'LATENT_DIM':    parser.getint(target_section, 'LATENT_DIM'),
        'NUM_EPOCHS':    parser.getint(target_section, 'NUM_EPOCHS'),
        'VARKEY':        parser.get(target_section, 'VARKEY'),
        'UPSAMPLING':    parser.get(target_section, 'UPSAMPLING'),
        'FNAME_TRAIN':   parser.get_required_option(target_section, 'FNAME_TRAIN'),
        'FNAME_VAL':     parser.get_required_option(target_section, 'FNAME_VAL'),
        'TARGET_SECTION': target_section,
    }

def main(config):
    fname_train   = config['FNAME_TRAIN'] 
    fname_val     = config['FNAME_VAL'] 
    fname_model   = Path("output") / config['TARGET_SECTION'] / "model.pt"
    fname_history = Path("output") / config['TARGET_SECTION'] / "history.csv"

    # Automatically creates folders if they don't exist
    fname_model.parent.mkdir(parents=True, exist_ok=True)

    varkey = config['VARKEY']

    ## 1. Loading raw data and normaliation
    ds1 = xr.open_dataset(fname_train)
    da1 = ds1[varkey]
    ds2 = xr.open_dataset(fname_val)
    da2 = ds2[varkey]

    ## 2. Create a custom Dataset
    min_value = config['MINVAL']
    max_value = config['MAXVAL']
    transform = MinMaxScale(min_value, max_value)

    train_dataset = EnsembleDataset(da1, transform)
    val_dataset   = EnsembleDataset(da2, transform)

    ## 3. Create a DataLoader
    train_loader = DataLoader(train_dataset, 
                              batch_size=config['BATCH_SIZE'], 
                              shuffle=True)
    val_loader   = DataLoader(val_dataset,
                              batch_size=config['BATCH_SIZE'],
                              shuffle=False)

    ## 4. Define a model
    model = VariationalAutoencoder(config['LATENT_DIM'], upsampling=config['UPSAMPLING'].lower())
#    summary(model, (1,101,121))

    ## 5. Loss function
    criterion = VAELoss(beta=config['BETA'])

    ## 6. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['LEARNING_RATE'])

    ## Training loop
    history = []

    for epoch in range(config['NUM_EPOCHS']):
        out_train = train_epoch(model, train_loader, criterion, optimizer)
        out_val   = evaluate_epoch(model, val_loader, criterion)
        # Store current losses
        history.append(out_train | out_val)
        if epoch%10 == 0 or epoch == config['NUM_EPOCHS']-1:
            print(f"-> Epoch {epoch+1:02d} \n"
                  f"   Train metrics {out_train} \n"
                  f"   Validation metrics: {out_val}")
    print("Done!")

    ## Save metrics
    df = pd.DataFrame(history)
    df.to_csv(fname_history, index=False)

    ## Save trained model
    torch.save({
        'model_state_dict': model.state_dict(),  # Trained Model parameters
        **config
        }, fname_model)

if __name__ == "__main__":
    # Input parameters and options
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS,description=__doc__)
    parser.add_argument('-s', '--section', 
                        default = None,
                        help='Section in configuration file', 
                        type=str)
    args = parser.parse_args()

    # Configuration
    config = load_config("config.ini", section=args.section)

    print_config(config)
    main(config)
