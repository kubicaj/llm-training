from torch.utils.data import Dataset, DataLoader


class TokenDataset(Dataset):
    """
    Class of token datasets
    """

    def __init__(self, tokens, seq_len=8, stride=4):
        # initialize
        self.inputs = []
        self.targets = []

        # overlapping sequences of seq_len
        for i in range(0, len(tokens) - seq_len, stride):
            # get c tokens and append to the lists
            self.inputs.append(tokens[i: i + seq_len])
            self.targets.append(tokens[i + 1: i + seq_len + 1])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
