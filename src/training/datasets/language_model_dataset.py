
from torch.utils.data import Dataset, DataLoader
from typing import List
import torch


class TextDataset(Dataset):

    def __init__(
            self,
            sequences: List[List[int]],
            seq_length: int = 50,
            stride: int = 25
    ):
        
        self.seq_length = seq_length
        self.stride = stride

        # Flatten all sequences
        self.data = []
        for seq in sequences:
            self.data.extend(seq)


        # Create sliding windows
        self.samples = []

        for i in range(0, len(self.data) - seq_length, stride):
            input_seq = self.data[i: i + seq_length]
            output_seq = self.data[i + 1: i + seq_length + 1]
            self.samples.append((input_seq, output_seq))


    def __len__(self):
        return len(self.samples)
    


    def __getitem__(self, index):

        input_seq, target_seq = self.samples[index]

        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(target_seq, dtype=torch.long)
        )