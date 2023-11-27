from torch.utils.data import random_split, Dataset, DataLoader
import pytorch_lightning as pl
from typing import Optional
import torchvision
import numpy as np


class CIFAR_10_Dataset(Dataset):

    def __init__(self, root_dir, train=True):
        """
        Arguments:
            root_dir (string): Directory with all the images.
        """
        self.root_dir = root_dir

        transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(args.img_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        self.dataset = torchvision.datasets.CIFAR10(root='./CIFAR_10_data', train=train,
                                        download=True, transform=transform)
            


    @staticmethod
    def generate_mask(num_players: int, num_mask_samples: int or None = None, paired_mask_samples: bool = True,
                  mode: str = 'uniform', random_state: np.random.RandomState or None = None) -> np.array:
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
        random_state = random_state or np.random

        num_samples_ = num_mask_samples or 1

        if paired_mask_samples:
            assert num_samples_ % 2 == 0, "'num_samples' must be a multiple of 2 if 'paired' is True"
            num_samples_ = num_samples_ // 2
        else:
            num_samples_ = num_samples_

        if mode == 'uniform':
            masks = (random_state.rand(num_samples_, num_players) > random_state.rand(num_samples_, 1)).astype('int')
        elif mode == 'shapley':
            probs = 1 / (np.arange(1, num_players) * (num_players - np.arange(1, num_players)))
            probs = probs / probs.sum()
            masks = (random_state.rand(num_samples_, num_players) > 1 / num_players * random_state.choice(
                np.arange(num_players - 1), p=probs, size=[num_samples_, 1])).astype('int')
        else:
            raise ValueError("'mode' must be 'random' or 'shapley'")

        if paired_mask_samples:
            masks = np.stack([masks, 1 - masks], axis=1).reshape(num_samples_ * 2, num_players)

        if num_mask_samples is None:
            masks = masks.squeeze(0)
            return masks  # (num_masks)
        else:
            return masks  # (num_samples, num_masks)


    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):

        image = self.dataset[idx]
        sample = {'image': image, 'masks': generate_mask(num_players=9, num_mask_samples=2)}

        return sample


class CIFAR_10_Datamodule(pl.LightningDataModule):

    def __init__(self,):
        super(CIFAR_10_Datamodule).__init__()
        
    def setup(self, stage: Optional[str] = None):
     
        if stage == "fit" or stage is None:
            train_set_full =  CIFAR_10_Dataset(
                root_path="./CIFAR_10_data",
                train=True)
            train_set_size = int(len(train_set_full) * 0.9)
            valid_set_size = len(train_set_full) - train_set_size
            self.train, self.validate = random_split(train_set_full, [train_set_size, valid_set_size])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = CIFAR_10_Dataset(
                root_path="./CIFAR_10_data",
                train=False)
            
    # define your dataloaders
    # again, here defined for train, validate and test, not for predict as the project is not there yet. 
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=32, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=32, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=32, num_workers=8)