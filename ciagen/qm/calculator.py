import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader


class MeanCalculator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._samples_computed = 0
        self._current_sum = 0

    def forward(self, x):
        with torch.no_grad():
            number_of_samples_in_x = x.shape[0]

            self._current_sum += torch.sum(x, axis=0)
            self._samples_computed += number_of_samples_in_x

        return True

    def get_mean(self):
        current_mean = self._current_sum / self._samples_computed
        return current_mean


class CovCalculator(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self._mean_calculator = MeanCalculator()

        self._samples_computed = 0
        self._cov_accum = 0

    def forward(self, x):
        self._mean_calculator(x)
        with torch.no_grad():
            number_of_samples_in_x = x.shape[0]

            # torch recommends using mT to transpose batches of matrices
            self._cov_accum += torch.matmul(x.mT, x)
            self._samples_computed += number_of_samples_in_x

        return True

    def get_cov(self):
        current_mean = self._mean_calculator.get_mean().unsqueeze(0)

        # print(self._cov_accum.size())
        # print(current_mean.size())
        current_cov = self._cov_accum - self._samples_computed * torch.matmul(
            current_mean.T, current_mean
        )

        current_cov = current_cov / (self._samples_computed - 1)  # unbiased estimator

        return current_cov


class CDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    a = MeanCalculator()
    tt = torch.rand((100, 10))

    print(torch.cov(tt.T), torch.cov(tt.T).size())

    tdataset = CDataset(tt)
    tdataloader = DataLoader(tdataset, batch_size=2)

    t = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

    b = CovCalculator()
    for x in tdataloader:
        b(x)

    # ee = b.get_cov()
    print(b.get_cov(), b.get_cov().size())
