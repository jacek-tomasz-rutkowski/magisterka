from typing import Optional
from pathlib import Path

import pytorch_lightning as pl
import torchvision
import numpy as np
from torch.utils.data import random_split, Dataset, DataLoader


class CIFAR_10_Dataset(Dataset):
    """CIFAR-10 dataset, but returns normalized tensors of shape (224, 224, 3)."""
    def __init__(self, num_players: int, num_mask_samples: int, paired_mask_samples: bool,
                 root_path: Path, train: bool = True, download: bool = True):
        """
        Args:
            root_path: Directory with all the images (will be downloaded if not present).
        """
        self.num_players=num_players
        self.num_mask_samples=num_mask_samples
        self.paired_mask_samples=paired_mask_samples
        self.root_path = root_path
        self.shape = (224, 224, 3)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.shape[:2]),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.dataset = torchvision.datasets.CIFAR10(
            root=root_path,
            train=train,
            download=download,
            transform=transform
        )

    @staticmethod
    def generate_mask(
            num_players: int, num_mask_samples: Optional[int] = None, paired_mask_samples: bool = True,
            mode: str = 'uniform', random_state: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Args:
            num_players: the number of players in the coalitional game
            num_mask_samples: the number of masks to generate
            paired_mask_samples: if True, the generated masks are pairs of x and 1-x.
            mode: the distribution that the number of masked features follows. ('uniform' or 'shapley')
            random_state: random generator

        Returns:
            torch.Tensor of shape
            (num_masks, num_players) if num_masks is int
            (num_players) if num_masks is None

        """
        random_state = random_state or np.random.default_rng()

        num_samples_ = num_mask_samples or 1

        if paired_mask_samples:
            assert num_samples_ % 2 == 0, "'num_samples' must be a multiple of 2 if 'paired' is True"
            num_samples_ = num_samples_ // 2
        else:
            num_samples_ = num_samples_

        if mode == 'uniform':
            thresholds = random_state.random((num_samples_, 1))
            masks = (random_state.random((num_samples_, num_players)) > thresholds).astype('int')
        elif mode == 'shapley':
            probs = 1 / (np.arange(1, num_players) * (num_players - np.arange(1, num_players)))
            probs = probs / probs.sum()
            thresholds = random_state.choice(np.arange(num_players - 1), p=probs, size=(num_samples_, 1))
            thresholds /= num_players
            masks = (random_state.random((num_samples_, num_players)) > thresholds).astype('int')
        else:
            raise ValueError("'mode' must be 'random' or 'shapley'")

        if paired_mask_samples:
            masks = np.stack([masks, 1 - masks], axis=1).reshape(num_samples_ * 2, num_players)

        if num_mask_samples is None:
            masks = masks.squeeze(0)
            return masks  # (num_masks)
        else:
            return masks  # (num_samples, num_masks)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        image, label = self.dataset[idx]
        sample = {'images': image, 'labels': label, 'masks': 
                  self.generate_mask(num_players=self.num_players, 
                                     num_mask_samples=self.num_mask_samples,
                                     paired_mask_samples=self.paired_mask_samples)}
        return sample


class CIFAR_10_Datamodule(pl.LightningDataModule):

    def __init__(self, num_players,
                 num_mask_samples,
                 paired_mask_samples):
        self.root_path = Path("./CIFAR_10_data").resolve()
        super().__init__()
        self.prepare_data_per_node = True
        self.num_players=num_players
        self.num_mask_samples=num_mask_samples
        self.paired_mask_samples=paired_mask_samples

    def prepare_data(self):
        # download
        CIFAR_10_Dataset(
            num_players=self.num_players,
            num_mask_samples=self.num_mask_samples,
            paired_mask_samples=self.paired_mask_samples,
            root_path=self.root_path,
            train=True,
            download=True)
        CIFAR_10_Dataset(
            num_players=self.num_players,
            num_mask_samples=self.num_mask_samples,
            paired_mask_samples=self.paired_mask_samples,
            root_path=self.root_path,
            train=False,
            download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            train_set_full = CIFAR_10_Dataset(num_players=self.num_players,
                                    num_mask_samples=self.num_mask_samples,
                                    paired_mask_samples=self.paired_mask_samples,
                                    root_path=self.root_path, 
                                    train=True)
            train_set_size = int(len(train_set_full) * 0.9)
            valid_set_size = len(train_set_full) - train_set_size
            self.train, self.validate = random_split(train_set_full, [train_set_size, valid_set_size])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = CIFAR_10_Dataset(num_players=self.num_players,
                                    num_mask_samples=self.num_mask_samples,
                                    paired_mask_samples=self.paired_mask_samples,
                                    root_path=self.root_path, 
                                    train=False)

    # define your dataloaders
    # again, here defined for train, validate and test, not for predict as the project is not there yet.
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=32, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=32, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=32, num_workers=0)
