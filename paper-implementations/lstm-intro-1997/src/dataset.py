import torch
from torch.utils.data import Dataset, DataLoader
import os

#-------------------#
# Character Dataset #
#-------------------#
class CharDataset(Dataset):
    def __init__(self, text, seq_len, char2int, int2char):
        """
        text: raw string
        seq_len: sequence length
        char2int: shared mapping of char to index
        int2char: shared mapping of index to char
        """
        self.seq_len = seq_len
        self.char2int = char2int
        self.int2char = int2char
        self.vocab_size = len(char2int)

        # Covert entire text to integers
        self.input_ids = [self.char2int[ch] for ch in text]

    def __len__(self):
        # Possible no of samples
        return len(self.input_ids) - self.seq_len
    
    def __getitem__(self, idx):
        # x: sequence, y: same sequence shifted by 1
        chunk = self.input_ids[idx : idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

#------------------#
# Get Data Loaders #
#------------------#
def get_dataloaders(seq_len, batch_size):
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")

    # Read all files
    def read_file(filename):
        with open(os.path.join(DATA_DIR, filename), "r", encoding="utf-8") as f:
            return f.read()

    train_text = read_file("train.txt")
    val_text   = read_file("val.txt")
    test_text  = read_file("test.txt")
    toy_text   = read_file("toy.txt")

    # Build shared vocabulary
    all_text = train_text + val_text + test_text + toy_text
    chars = sorted(list(set(all_text)))
    char2int = {ch: i for i, ch in enumerate(chars)}
    int2char = {i: ch for i, ch in enumerate(chars)}
    vocab_size = len(chars)

    #-----------------#
    # Create Datasets #
    #-----------------#
    toy_dataset = CharDataset(toy_text, seq_len, char2int, int2char)
    train_dataset = CharDataset(train_text, seq_len, char2int, int2char)
    val_dataset = CharDataset(val_text, seq_len, char2int, int2char)
    test_dataset = CharDataset(test_text, seq_len, char2int, int2char)
    
    #--------------------#
    # Create Dataloaders #
    #--------------------#
    toy_loader = DataLoader(toy_dataset, batch_size, shuffle=True, drop_last=True)   
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, drop_last=False)

    return toy_loader, train_loader, val_loader, test_loader, vocab_size, char2int, int2char