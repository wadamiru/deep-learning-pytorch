from lstm import LSTMWrapper
from dataset import CharDataset
from torch.utils.data import DataLoader
from train import train_model
import os

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
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

train_path = os.path.join(DATA_DIR, "train.txt")
toy_path = os.path.join(DATA_DIR, "toy.txt")

with open(train_path, "r", encoding="utf-8") as f:
    train_text = f.read()

with open(toy_path, "r", encoding="utf-8") as f:
    toy_text = f.read()

dataset = CharDataset(toy_text, SEQ_LEN)
loader = DataLoader(dataset, BATCH_SIZE, shuffle=True, drop_last=True)

#-------------#
# MODEL SETUP #
#-------------#
model = LSTMWrapper(dataset.vocab_size, HIDDEN_SIZE, NUM_LAYERS)

#-------------#
# TRAIN MODEL #
#-------------#
train_model(model, loader, EPOCHS, LR)