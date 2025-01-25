from torch.utils.data import Subset, Dataset

class Partitioner:

    def partition(self, partitioning_method: str, dataset: Subset) -> dict[int, list[int]]:
        """
        Splits a torch Subset following a given method.
        Implemented methods for label skewness are: IID, Hard, Dirichlet
        :param partitioning_method: a string containing the name of the partitioning method.
        :param dataset: a torch Subset containing the dataset to be partitioned.
        :return: a dict in which keys are the IDs of the subareas and the values are lists of IDs of the instances of the subarea
            (IDs references the original dataset).
        """
        pass

    def download_dataset(self, dataset_name: str, train: bool = True) -> Dataset:
        """
        Download the specified dataset from torchvision.
        Valid datasets are: MNIST, FashionMNIST, Extended MNIST, CIFAR-10, CIFAR-100.
        :param dataset_name: The dataset to be downloaded.
        :param train: Whether to download the training set or the test set.
        :return: the specified dataset.
        """
        pass

    def train_validation_split(self, train_percentage: float) -> tuple[Subset, Subset]:
        """
        Split a given dataset in training and validation set.
        :param train_percentage: The percentage of training instances, it must be a value between 0 and 1.
        :return: A tuple containing the training and validation subsets.
        """
        pass

    def __partition_hard(self):
        pass

    def _partition_iid(self):
        pass

    def __partition_dirichlet(self):
        pass