import torch
from transformers import SwinConfig, \
    SwinForImageClassification, ViTForImageClassification

from models.t2t_vit import t2t_vit_14
from vit_shapley.CIFAR_10_Dataset import PROJECT_ROOT
from vit_shapley.modules.surrogate import Surrogate

def load_vit(target_model_path="saved_models/transferred/cifar10/vit_epoch-47_acc-98.2.pth"):
    target_model_vit = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", 
                                                                 num_labels=10,
                                                                 ignore_mismatched_sizes=True)

    target_model_path_vit = PROJECT_ROOT / target_model_path
    state_dict = torch.load(target_model_path_vit)
    target_model_vit.load_state_dict(state_dict, strict=False)
    return target_model_vit

def load_swin(target_model_path="saved_models/transferred/cifar10/swin_epoch-37_acc-97.34.pth")
    configuration = SwinConfig()
    target_model_swin = SwinForImageClassification(configuration) \
                                    .from_pretrained("microsoft/swin-tiny-patch4-window7-224",\
                                    num_labels=10,
                                    ignore_mismatched_sizes=True)

    target_model_path_swin = PROJECT_ROOT / target_model_path
    state_dict = torch.load(target_model_path_swin)
    target_model_swin.load_state_dict(state_dict, strict=False)
    return target_model_swin

def load_t2t(target_model_path="saved_models/transferred/cifar10/ckpt_0.01_0.0005_97.5.pth"):
    target_model_t2t = models.t2t_vit.t2t_vit_14(num_classes=10)
    target_model_path_t2t = PROJECT_ROOT / target_model_path
    state_dict = torch.load(target_model_path_t2t)
    target_model_t2t.load_state_dict(state_dict, strict=False)

def load_surrogate(backbone_name:str, surrogate_path:str):
    surrogate_t2t_vit = Surrogate.load_from_checkpoint(
    PROJECT_ROOT / surrogate_path,
    backbone_name=backbone_name,
    strict=False)
    return surrogate_t2t_vit

