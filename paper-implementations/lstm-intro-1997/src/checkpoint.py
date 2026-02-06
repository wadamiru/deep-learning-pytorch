#----------------------#
# Save/Load the models #
#----------------------#

import torch
import os

def save_best(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load(model, path, device):
    ## Load a pre-saved model
    model.load_state_dict(torch.load(path, map_location=device))