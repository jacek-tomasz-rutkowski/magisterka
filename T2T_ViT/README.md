## saved_models
Contains checkpoints with model weights that we want to keep:
- `downloaded/`: downloaded from https://github.com/yitu-opensource/T2T-ViT
- `transferred/cifar10/`: transferred from downloaded model to cifar10 dataset, trained with something like
    `python transfer_learning.py --model t2t_vit_14 --lr 0.05 --b 64 --num-classes 10 --img-size 224 --transfer-learning True --transfer-model saved_models/downloaded/imagenet/81.5_T2T_ViT_14.pth`
- `surrogate/cifar10/`: surrogate, trained with something like
    `python -m vit_shapley.modules.surrogate --num_players 196 --lr 0.00001 --wd 0 --b 256 --num_workers 2`