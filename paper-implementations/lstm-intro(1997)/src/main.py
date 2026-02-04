from lstm import LSTMWrapper
from dataset import CharDataset
from torch.utils.data import DataLoader
from train import train_model

#---------------#
# CONFIGURATION #
#---------------#
SEQ_LEN = 50
HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS = 2
LR = 0.002
EPOCHS = 100

#-----------#
# LOAD DATA #
#-----------#
text = ""
dataset = CharDataset(text, SEQ_LEN)
loader = DataLoader(dataset, BATCH_SIZE, shuffle=True, drop_last=True)

#-------------#
# MODEL SETUP #
#-------------#
model = LSTMWrapper(dataset.vocab_size, HIDDEN_SIZE, NUM_LAYERS)

#-------------#
# TRAIN MODEL #
#-------------#
train_model(model, loader, EPOCHS, LR)