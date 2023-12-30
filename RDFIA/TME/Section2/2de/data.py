from typing import Any

import ignite.distributed as idist
import torchvision
import torchvision.transforms as T


def setup_data(config: Any):
    """Download datasets and create dataloaders

    Parameters
    ----------
    config: needs to contain `data_path`, `batch_size`, `eval_batch_size`, and `num_workers`
    """
    mytransform = T.Compose([T.Resize(32), T.ToTensor(), T.Normalize((0.5), (0.5))])

    dataset_train = torchvision.datasets.MNIST(
        root=config.data_path, download=True, train=True, transform=mytransform
    )
    dataset_eval = torchvision.datasets.MNIST(
        root=config.data_path,
        train=False,
        download=True,
        transform=mytransform,
    )

    # dataset_train = torchvision.datasets.CIFAR10(
    #     root=config.data_path, download=True, train=True, transform=mytransform
    # )
    # dataset_eval = torchvision.datasets.CIFAR10(
    #     root=config.data_path,
    #     train=False,
    #     download=True,
    #     transform=mytransform,
    # )

    # dataset_train = torchvision.datasets.CelebA(
    #     root=config.data_path, download=True, split="train", transform=mytransform
    # )
    # dataset_eval = torchvision.datasets.MNIST(
    #     root=config.data_path,
    #     split="test",
    #     download=True,
    #     transform=mytransform,
    # )

    dataloader_train = idist.auto_dataloader(
        dataset_train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    dataloader_eval = idist.auto_dataloader(
        dataset_eval,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    nc = 1  # num channels
    # nc = 3  # CelebA, CIFA10 num channels
    return dataloader_train, dataloader_eval, nc
