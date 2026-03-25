import torch
from torch.utils.data import Dataset


class torchDataset(Dataset):

    def __init__(self, dataset):
        # Feature matrix
        self.x = torch.from_numpy(dataset['x']).float()
        # Outcome (LoanApproved / acceptance)
        self.y = torch.from_numpy(dataset['y']).float()
        # Dosage (normalized interest rate)
        self.d = torch.from_numpy(dataset['d']).float()

        self.length = self.x.shape[0]

    def get_data(self):
        return self.x, self.y, self.d

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.d[index]

    def __len__(self):
        return self.length
