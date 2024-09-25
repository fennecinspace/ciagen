import torch
from abc import ABC


class ABCAccum(ABC, torch.nn.Module):

    @property
    def _accums(self):
        accums = [x for x in self.__dict__ if x.endswith("_accum")]
        return accums

    @property
    def _composed_accums(self):
        accums = [x for x in self.__dict__ if isinstance(x, ABCAccum)]
        return accums

    def reset(self):
        for accum in self._accums:
            setattr(self, accum, None)

    def verify_integrity(self):
        accums = self._accums
        caccums = self._composed_accums

        first_value = getattr(self, accums[0])
        first_value_is_none = first_value is None

        for accum in accums[1:]:
            value = getattr(self, accum)
            value_is_none = value is None
            if first_value_is_none and not value_is_none:
                return False
            if not first_value_is_none and value_is_none:
                return False

        for caccum in caccums:
            value = getattr(self, caccum)
            if value.verify_integrity() is False:
                return False

        return True

    def forward(self, x):
        if not self.verify_integrity():
            raise RuntimeError("Accumulator is not in a valid state.")

        accums = self._accums
        caccums = self._composed_accums

        for caccum in caccums:
            value = getattr(self, caccum)
            value.forward(x)

        for accum in accums:
            accum_updater = getattr(self, f"{accum}_update")
            accum_updater(x)

        return self.state()

    # TODO: finish this function
    # def state(self):
    #     accums = self._accums
    #     caccums = self._composed_accums

    #     for caccum in caccums:
    #         value = getattr(self, caccum)
    #         value_state = value.state()

    #     state = {}
    #     for accum in accums:
    #         value = getattr(self, accum)
    #         value_state = value.state()

    #     pass


class MeanCalculator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._samples_computed = None
        self._current_sum = None

    def reset(self):
        self._samples_computed = None
        self._current_sum = None

    def _verify_integrity(self):
        return (self._samples_computed is None and self._current_sum is None) or (
            self._samples_computed is not None and self._current_sum is not None
        )

    def forward(self, x):
        if not self._verify_integrity():
            raise RuntimeError("MeanCalculator is not in a valid state.")

        with torch.no_grad():
            number_of_samples_in_x = x.shape[0]

            if self._current_sum is None:
                self._current_sum = torch.sum(x, axis=0)
                self._samples_computed = number_of_samples_in_x

            else:
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

        self._samples_computed = None
        self._cov_accum = None

    def reset(self):
        self._samples_computed = None
        self._cov_accum = None

        self._mean_calculator.reset()

    def _verify_integrity(self):
        if self._mean_calculator._verify_integrity() is False:
            return False

        return (self._samples_computed is None and self._cov_accum is None) or (
            self._samples_computed is not None and self._cov_accum is not None
        )

    def forward(self, x):
        if not self._verify_integrity():
            raise RuntimeError("CovCalculator is not in a valid state.")

        with torch.no_grad():
            self._mean_calculator(x)

            number_of_samples_in_x = x.shape[0]

            # torch recommends using mT to transpose batches of matrices
            if self._cov_accum is None:
                self._cov_accum = torch.matmul(x.mT, x)
                self._samples_computed = number_of_samples_in_x
            else:
                self._cov_accum += torch.matmul(x.mT, x)
                self._samples_computed += number_of_samples_in_x

        return self.state()

    def state(self):
        current_mean = self._mean_calculator.state().unsqueeze(0)

        mean_matrix = torch.matmul(current_mean.T, current_mean)
        current_cov = self._cov_accum - self._samples_computed * mean_matrix

        current_cov = current_cov / (self._samples_computed - 1)  # unbiased estimator

        return current_cov


class KLCalculator(torch.nn.Module):
    def __init__(self, eps=1e-16):
        super().__init__()
        self._eps = eps

        self._samples_computed = None
        self._conditional_probability_accumulator = None
        self._total_probability_accumulator = None

    def reset(self):
        self._samples_computed = None
        self._conditional_probability_accumulator = None
        self._total_probability_accumulator = None

    def _verify_integrity(self):
        return (
            self._samples_computed is None
            and self._conditional_probability_accumulator is None
        ) or (
            self._samples_computed is not None
            and self._conditional_probability_accumulator is not None
        )

    def forward(self, x):
        if not self._verify_integrity():
            raise RuntimeError("KLCalculator is not in a valid state.")

        with torch.no_grad():
            number_of_samples_in_x = x.shape[0]
            conditional_accumulator = torch.sum(x * torch.log(x + self._eps), dim=0)
            total_accumulator = torch.sum(x, dim=0)

            if self._samples_computed is None:
                self._samples_computed = number_of_samples_in_x

                self._conditional_probability_accumulator = conditional_accumulator
                self._total_probability_accumulator = total_accumulator
            else:
                self._samples_computed += number_of_samples_in_x
                self._conditional_probability_accumulator += conditional_accumulator
                self._total_probability_accumulator += total_accumulator

        return self.state()

    def state(self):
        current_conditional_probability = self._conditional_probability_accumulator
        current_total_probability = self._total_probability_accumulator
        current_number_of_samples = self._samples_computed

        total_probability = current_total_probability / current_number_of_samples
        total_probability_log = torch.log(total_probability + self._eps)

        # This is the KL divergence
        kl_difference = (
            current_conditional_probability
            - current_total_probability * total_probability_log
        )

        # Now we compute its expectation plus exp
        expectation_kl = torch.sum(kl_difference) / current_number_of_samples
        return torch.exp(expectation_kl)


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
