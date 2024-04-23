## saved_models/
Contains checkpoints with model weights that we want to keep:
- `downloaded/`: downloaded from https://github.com/yitu-opensource/T2T-ViT
- `transferred/cifar10/`:
    - `ckpt_0.01_0.0005_97.5.pth`: T2T-ViT, accuracy 97.5% without masks, 68.58% with 196-masks. Trained with something like:
        `python transfer_learning.py --model t2t_vit_14 --lr 0.05 --b 64 --num-classes 10 --img-size 224 --transfer-learning True --transfer-model saved_models/downloaded/imagenet/81.5_T2T_ViT_14.pth`
    - `swin_epoch-37_acc-97.34.pth`: Swin Transformer, accuracy 97.34% without masks, 75.53% with 196-masks. Trained with:
        `python transfer_learning_swin.py ??`
    - `vit_epoch-47_acc-98.2.pth`: plain ViT, accuracy 98.20% without masks, 75.77% with 196-masks. Trained with:
        ???
- `surrogate/cifar10/`:
  - `_t2t_vit_player16_lr0.0001_wd0.0_b256_epoch19.ckpt`: accuracy 97.97% without masks, 85.25% with 196-masks (TODO improve).
  - `_swin_player16_lr0.0001_wd0.0_b256_epoch19.ckpt`: accuracy 98.05% without masks, 91.54% with 196-masks.
  - `_vit_player16_lr0.0001_wd0.0_b256_epoch19.ckpt`: accuracy 98.72% without masks, 90.17% with 196-masks.
  - `_player16_lr1e-05_wd0.0_b256_epoch28.ckpt`: old T2T-ViT, accuracy 98.29% without masks, 91.49% with 196-masks. Trained with something like:
        `python -m vit_shapley.modules.surrogate --num_players 196 --lr 0.00001 --wd 0 --b 256 --num_workers 2`
