from lstm import LSTMWrapper
from dataset import get_dataloaders
from early_stopping import EarlyStopping
import torch
from torch import optim, nn
from train import train_model

#---------------#
# CONFIGURATION #
#---------------#
SEQ_LEN = 50
HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS = 1
LR = 0.01
EPOCHS = 100

#-----------#
# LOAD DATA #
#-----------#
toy_loader, train_loader, val_loader, test_loader, vocab_size, _, _ = get_dataloaders(SEQ_LEN, BATCH_SIZE)

#-------------#
# MODEL SETUP #
#-------------#
model = LSTMWrapper(vocab_size, HIDDEN_SIZE, NUM_LAYERS)

#-------------#
# TRAIN MODEL #
#-------------#
optimiser = optim.Adam(model.parameters(), LR)
criterion = nn.CrossEntropyLoss()
early_stopper = EarlyStopping(patience=5, min_delta=0.01)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_model(model, train_loader, val_loader, optimiser, criterion, early_stopper, device, EPOCHS)