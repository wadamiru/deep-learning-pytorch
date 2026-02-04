from lstm import DeepLSTM
from dataset import CharDataset
from torch.utils.data import DataLoader

#---------------#
# CONFIGURATION #
#---------------#
SEQ_LEN = 50
HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_LAYERS = 2
LR = 0.002

#-----------#
# LOAD DATA #
#-----------#
text = ""
dataset = CharDataset(text, SEQ_LEN)
loader = DataLoader(dataset, BATCH_SIZE, shuffle=True, drop_last=True)

