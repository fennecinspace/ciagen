import torch


class MeanCalculator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._samples_computed = 0
        self._current_sum = 0

    def reset(self):
        self._samples_computed = 0
        self._current_sum = 0

    def forward(self, x):
        with torch.no_grad():
            number_of_samples_in_x = x.shape[0]

            self._current_sum += torch.sum(x, axis=0)
            self._samples_computed += number_of_samples_in_x

        return self.state()

    def state(self):
        current_mean = self._current_sum / self._samples_computed
        return current_mean


class CovCalculator(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self._mean_calculator = MeanCalculator()

        self._samples_computed = 0
        self._cov_accum = 0

    def reset(self):
        self._samples_computed = 0
        self._cov_accum = 0

        self._mean_calculator.reset()

    def forward(self, x):
        with torch.no_grad():
            self._mean_calculator(x)

            number_of_samples_in_x = x.shape[0]

            # torch recommends using mT to transpose batches of matrices
            self._cov_accum += torch.matmul(x.mT, x)
            self._samples_computed += number_of_samples_in_x

        return self.state()

    def state(self):
        current_mean = self._mean_calculator.get_mean().unsqueeze(0)

        mean_matrix = torch.matmul(current_mean.T, current_mean)
        current_cov = self._cov_accum - self._samples_computed * mean_matrix

        current_cov = current_cov / (self._samples_computed - 1)  # unbiased estimator

        return current_cov


if __name__ == "__main__":
    from torch.utils.data.dataloader import DataLoader
    from tqdm import tqdm

    class CDataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __getitem__(self, index):
            return self.dataset[index]

        def __len__(self):
            return len(self.dataset)

    torch.set_printoptions(precision=8)
    m = MeanCalculator()
    c = CovCalculator()

    for i in range(10):
        m.reset()
        c.reset()

        tt = torch.rand((2000, 768))

        tdataset = CDataset(tt)
        tdataloader = DataLoader(tdataset, batch_size=32)
        print("finished creating dataset")

        for x in tqdm(tdataloader):
            m(x)
            c(x)

        print("=======================================")
