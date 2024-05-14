import torch
import scipy.special
from tqdm import tqdm

import models.t2t_vit
from datasets.CIFAR_10_Dataset import PROJECT_ROOT, CIFAR_10_Datamodule, apply_masks_to_batch
from vit_shapley.modules.surrogate import Surrogate

class Shap_values():
    def __init__(self, model: torch.nn.Module,
                 images: torch.Tensor,
                 targets: torch.Tensor,
                 num_players: int,
                 num_classes: int=10):
        super(Shap_values, self).__init__()
        self.num_players = num_players
        self.images = images
        self.targets = targets
        self.model = model
        self.num_classes = num_classes
        self.batch_size = images.shape[0]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_all_masks(self, K) -> torch.Tensor:
        """
        K is equal to num_players or to num_players or num_players- 1
        returns all 0-1 masks shape (2**K, K)
        and expands to shape (B, 2**K, K)
        """
        K, N = K, 1
        masks = (torch.bitwise_and (torch.arange (2**(K*N)).unsqueeze (-1), \
                            2**torch.arange (K*N)) > 0).long().\
                                reshape (2**(K*N), K, N)
        return masks.squeeze().expand(self.batch_size,-1,-1)


    def compute_model_on_patches(self, masks: torch.Tensor,
                                 ) -> torch.Tensor:
        """
        masks is tensor of shape (B, 2**num_players, num_players),
        we want to compute values of the model for every
        image and mask
        image is tensor of shape (B, C, H, W) - batch of images from the dataset
        returns: values of the model on images of size (B*(2**num_players), C, H, W),
        shape of output is (B, 2**num_players, num_classes)
        """

        # batch version
        max_dim = masks.shape[1]
        values = []
        self.model.eval()

        max_batch = 256//self.batch_size
        for i in tqdm(range(len(masks[1])//max_batch), desc='Compute model values'):
            masks_ = masks[:, max_batch*i: min(max_batch*(i+1), max_dim)]
            pictures_, _, _ = apply_masks_to_batch(images=self.images.to(self.device),
                                masks=masks_.to(self.device))
            masks_.cpu().detach()
            value = self.model(pictures_).cpu().detach().view(self.batch_size, -1, self.num_classes)
            # (B, max_batch, num_classes)
            # we have to resize before appending
            # for later concatenation
            values.append(value)

        return torch.cat(values, dim=1)
        


    def compute_dif_for_feature(self, model_values: torch.Tensor, player_number: int):
        
        masks = self.get_all_masks(K=self.num_players)
        # (B, 2**num_players, num_players)

        player_mask = (masks[:, :, player_number] == 1).flatten(0,1)
        # (B*2**num_players,)

        other_coalitions_mask = ~(player_mask)

        model_values = model_values.view(-1, model_values.shape[-1])
        # (b*2**num_players, num_players)

        dif = model_values[player_mask] - model_values[other_coalitions_mask]
        # (B*2**(num_players - 1), num_classes)

        return dif.view(self.batch_size, -1, self.num_classes)

    def shap_values(self):
        """
        computes shapley values from definition for batch of images
        and given feature number
        """
        almost_all_masks = self.get_all_masks(self.num_players - 1)
        # (B, 2*(num_players - 1), num_players - 1)

        covered = torch.sum(almost_all_masks, dim=-1)
        # (B, 2*(num_players - 1))

        shap_vs = torch.zeros(self.batch_size, self.num_players, self.num_classes)

        all_masks = self.get_all_masks(self.num_players)
        # (B, 2*(num_players), num_players)
        model_values = self.compute_model_on_patches(all_masks)
        # (B, 2**num_players, num_classes)

        values = []
        for player_number in tqdm(range(self.num_players), desc='compute shapley values'):
            values = self.compute_dif_for_feature(model_values, player_number)
            # (B*2**(num_players - 1), num_classes)

            for i in range(values.shape[1]):
                shap_vs[:, player_number] += values[:, i]/scipy.special.binom(self.num_players-1, covered[:, i].unsqueeze(1))

        return shap_vs/self.num_players


def main() -> None:
    # target_model = models.t2t_vit.t2t_vit_14(num_classes=10)
    # # target_model_path = PROJECT_ROOT / "saved_models/downloaded/cifar10/cifar10_t2t-vit_14_98.3.pth"
    # target_model_path = PROJECT_ROOT / "saved_models/transferred/cifar10/ckpt_0.01_0.0005_97.5.pth"
    # load_checkpoint(target_model_path, target_model, device='cuda')
    target_model = models.t2t_vit.t2t_vit_14(num_classes=10)
    # target_model_path = PROJECT_ROOT / "saved_models/transferred/cifar10/ckpt_0.01_0.0005_97.5.pth"
    # load_checkpoint(target_model_path, target_model)

    surrogate = Surrogate.load_from_checkpoint(
        PROJECT_ROOT / "saved_models/surrogate/cifar10/_player16_lr1e-05_wd0.0_b256_epoch28.ckpt",
        # PROJECT_ROOT / "saved_models/surrogate/cifar10/_player196_lr1e-05_wd0.0_b128_epoch49.ckpt",
        target_model=target_model,
        num_players=16
    )


    datamodule = CIFAR_10_Datamodule(num_players=16,
                                     num_mask_samples=1,
                                     paired_mask_samples=False,
                                     batch_size=3,
                                     num_workers=0)
    datamodule.setup()
    data = next(iter(datamodule.test_dataloader()))

    images = data['images']
    labels = data['labels']
    import time

    time_0 = time.time()

    Shap_vs = Shap_values(surrogate, images, labels,
                          num_players=16)

    print(Shap_vs.shap_values())

    print(f'time of computation is {time.time() - time_0} seconds')

if __name__ == '__main__':
    main()
