import torch
from torch.utils.data import Dataset, DataLoader

class CharDataset(Dataset):
    def __init__(self, text, seq_len):
        # Create a sorted list of unique elements
        self.chars = sorted(list(set(text)))
        self.char2int = {ch: i for i, ch in enumerate(self.chars)}
        self.int2char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.seq_len = seq_len

        # Covert entire text to integers
        self.input_ids = [self.char2int[ch] for ch in text]

    def __len__(self):
        # Possible no of samples
        return len(self.input_ids) - self.seq_len
    
    def __getitem__(self, idx):
        # x: sequence (e.g. "hell")
        # y: same sequence shifted by 1 (e.g. "ello")
        chunk = self.input_ids[idx : idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y