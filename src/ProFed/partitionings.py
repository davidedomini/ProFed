from torch.utils.data import Subset, Dataset
from torchvision import datasets, transforms

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

    def download_dataset(self, dataset_name: str, train: bool = True, transform: transforms.Compose = None, download_path: str = 'dataset') -> Dataset:
        """
        Download the specified dataset from torchvision.
        Valid datasets are: MNIST, FashionMNIST, Extended MNIST, CIFAR10, CIFAR100.
        :param dataset_name: The dataset to be downloaded.
        :param train: Whether to download the training set or the test set.
        :param transform: Transformations that will be applied to the dataset. If none only ToTensor will be applied.
        :param download_path: The path where the dataset will be downloaded.
        :return: the specified dataset.
        """
        if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])

        if dataset_name == 'MNIST':
            dataset = datasets.MNIST(root=download_path, train=train, download=True, transform=transform)
        elif dataset_name == 'CIFAR10':
            dataset = datasets.CIFAR10(root=download_path, train=train, download=True, transform=transform)
        elif dataset_name == 'CIFAR100':
            dataset = datasets.CIFAR100(root=download_path, train=train, download=True, transform=transform)
        elif dataset_name == 'EMNIST':
            dataset = datasets.EMNIST(root=download_path, split='letters', train=train, download=True, transform=transform)
        elif dataset_name == 'FashionMNIST':
            dataset = datasets.FashionMNIST(root=download_path, train=train, download=True, transform=transform)
        else:
            raise Exception(f'Dataset {dataset_name} not supported! Please check :)')
        return dataset

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