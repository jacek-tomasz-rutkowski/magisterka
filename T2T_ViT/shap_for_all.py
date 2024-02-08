import torch
import scipy.special
from tqdm import tqdm

import models.t2t_vit
from utils import load_checkpoint
from vit_shapley.CIFAR_10_Dataset import PROJECT_ROOT, CIFAR_10_Datamodule, apply_masks_to_batch


class Shap_values():
    def __init__(self, model: torch.nn.Module, 
                 images: torch.Tensor,
                 targets: torch.Tensor, 
                 num_players: int,
                 feature_number: int,
                 num_classes: int=10):
        super(Shap_values, self).__init__()
        self.num_players = num_players
        self.images = images
        self.targets = targets
        self.model = model
        self.feature_number = feature_number
        self.num_classes = num_classes
        self.batch_size = images.shape[0]
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def get_all_masks(self, K) -> torch.Tensor:
        """
        K is equal to num_players or to num_players - 1
        returns all 0-1 masks shape (2**num_players, num_players)
        and expands to shape (b, 2**num_players, num_players)
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
        we want to compute values of the model for each mask
        image is tensor of shape (B, C, H, W) - batch of images from the dataset
        returns: values of the model on images of size (B*(2**num_players), C, H, W),
        shape of output is (B*2**num_players, num_classes), 
        # after reshape it is (B, 2**num_players, num_classes)
        """
        
        # batch version
        max_dim = masks.shape[1]
        values = []
        self.model.eval()
        
        max_batch = 256//self.batch_size
        for i in tqdm(range(len(masks[0])//max_batch + 1), desc='Compute model values'):
            masks_ = masks[:, max_batch*i: min(max_batch*(i+1), max_dim)]
            pictures_, _, _ = apply_masks_to_batch(images=self.images.to(self.device), 
                                masks=masks_.to(self.device))
            masks_.cpu().detach()
            value = self.model(pictures_).cpu().detach()
            values.append(value)
        
        return torch.cat(values).view(self.batch_size, -1, self.num_classes)
    
    def compute_dif_for_feature(self, model_values: torch.Tensor, player_number: int):
        masks = self.get_all_masks(K=self.num_players)

        player_mask = masks[:, :, player_number] == 1
        other_coalitions_mask = masks[:, :, player_number] == 0 
        
        dif = model_values[player_mask] - model_values[other_coalitions_mask]
        
        return dif.view(self.batch_size, -1, self.num_classes)
        
    def shap_values(self, player_number: int):
        """
        computes shapley values from definition for batch of images
        and given feature number
        """
        almost_all_masks = self.get_all_masks(self.num_players - 1)
        covered = torch.sum(almost_all_masks, dim=-1)

        shap_values = torch.zeros(self.num_players, self.batch_size, self.num_classes)
        
        all_masks = self.get_all_masks(self.num_players)
        model_values = self.compute_model_on_patches(all_masks)
        values = []
        for player_number in range(self.num_players):
            values = self.compute_dif_for_feature(model_values, player_number)
        
            for i in tqdm(range(values.shape[1]), desc='compute shapley values'):
                shap_values[player_number] += values[:, i]/scipy.special.binom(self.num_players-1, covered[:, i].unsqueeze(1))
            
        print('Shapley values are:')
        return shap_values/self.num_players
    

def main() -> None:
    target_model = models.t2t_vit.t2t_vit_14(num_classes=10)
    # target_model_path = PROJECT_ROOT / "saved_models/downloaded/cifar10/cifar10_t2t-vit_14_98.3.pth"
    target_model_path = PROJECT_ROOT / "saved_models/transferred/cifar10/ckpt_0.01_0.0005_97.5.pth"
    load_checkpoint(target_model_path, target_model, device='cuda')

    datamodule = CIFAR_10_Datamodule(num_players=9,
                                     num_mask_samples=1, 
                                     paired_mask_samples=False,
                                     batch_size=3,
                                     num_workers=0)
    datamodule.setup()
    data = next(iter(datamodule.test_dataloader()))

    images = data['images']
    labels = data['labels']
    masks = data['masks']
    import time

    time_0 = time.time()

    Shap_vs = Shap_values(target_model, images, labels,
                          num_players=4, feature_number=3)

    print(Shap_vs.shap_values(player_number=3))

    print(f'time of computation is {time.time() - time_0} seconds')

if __name__ == '__main__':
    main()
