import torch
import torchvision
import math
from itertools import chain, combinations
import scipy.special
from tqdm import tqdm

import models.t2t_vit
from utils import load_checkpoint
from vit_shapley.CIFAR_10_Dataset import PROJECT_ROOT, CIFAR_10_Datamodule


class Shap_values():
    def __init__(self, model: torch.nn.Module, 
                 images: torch.Tensor, 
                 num_players: int, 
                 num_classes: int=10):
        super(Shap_values, self).__init__()
        self.num_players = num_players
        self.images = images
        self.model = model
        self.num_classes = num_classes
        self.device = images.device

    @staticmethod
    def powerset(a: list) -> list:
        return list(chain.from_iterable(combinations(a, r) for r in range(len(a)+1)))

    def fill_patches(self, patches: list, pictures: torch.Tensor) -> torch.Tensor:
        patch_size = pictures.shape[-1]//int(math.sqrt(self.num_players))
        masks = torch.zeros(pictures.shape).to(self.device)
        for patch in patches:
            vert, horz = patch
            masks = torchvision.transforms.functional.erase(masks,
                                                         i=vert*patch_size,
                                                         j=horz*patch_size,
                                                         h=patch_size,
                                                         w=patch_size,
                                                         v=1)
        
        return pictures.to(self.device)*masks
    
    def compute_model_on_patches(self, patches: list, pictures: torch.Tensor) -> torch.Tensor:
        pictures_ = self.fill_patches(patches, pictures)
        return self.model(pictures_)
    
    def values_list(self, images: torch.Tensor) ->torch.Tensor:
        patches = [(x,y) for x in range(int(math.sqrt(self.num_players)))\
                    for y in range(int(math.sqrt(self.num_players)))]
        values = []
        for regions in self.powerset(patches):
            values.append(self.compute_model_on_patches(regions, images))

        return values
        
    def compute_dif(self, regions: list, feature: list):
        filled = self.fill_patches(regions, self.images)
        filled_feature = self.fill_patches(feature, filled)

        dif = self.model(filled_feature) - self.model(filled)
        return dif

    def shap_values(self, feature):        
        patches = [(x,y) for x in range(int(math.sqrt(self.num_players)))\
                    for y in range(int(math.sqrt(self.num_players)))]
        patches.remove(feature[0]) # feature is a 1-elt list
        values = torch.zeros(len(self.images), self.num_classes).to(self.device)

        for i, regions in enumerate(tqdm(self.powerset(patches))):
            # print(f'it is {i}-th iteration')
            values += self.compute_dif(regions, feature)/\
                    scipy.special.binom(self.num_players-1, len(regions))
            print(f'values sum now is {torch.sum(values)}')
        return values/self.num_players
    

if __name__ == '__main__':
    target_model = models.t2t_vit.t2t_vit_14(num_classes=10)
    # target_model_path = PROJECT_ROOT / "saved_models/downloaded/cifar10/cifar10_t2t-vit_14_98.3.pth"
    target_model_path = PROJECT_ROOT / "saved_models/transferred/cifar10/ckpt_0.01_0.0005_97.5.pth"
    load_checkpoint(target_model_path, target_model)

    datamodule = CIFAR_10_Datamodule(num_players=9, 
                                     num_mask_samples=1, 
                                     paired_mask_samples=False,
                                     num_workers=0)
    datamodule.setup()
    data = next(iter(datamodule.test_dataloader()))

    images = data['images']
    labels = data['labels']
    masks = data['masks']
    import time

    time_0 = time.time()

    print('Loading finished')

    Shap_vs = Shap_values(target_model, images, num_players=9)
    print(Shap_vs.shap_values([(1,1)]))

    print(f'time of computation is {time.time() - time_0} seconds')
