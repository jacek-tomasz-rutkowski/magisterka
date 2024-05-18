# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.

"""Transfer pretrained T2T-ViT to downstream dataset: CIFAR10/CIFAR100."""
import argparse
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split

from utils import progress_bar
# from timm.models import create_model
from utils import load_for_transfer_learning
from datasets.gastro import GastroDataset, collate_gastro_batch

# Models used in --model need to be imported here.
# from models.t2t_vit import *
from models.t2t_vit import t2t_vit_14

PROJECT_ROOT = Path(__file__).parent

parser = argparse.ArgumentParser(description="PyTorch CIFAR10/CIFAR100 Training")
parser.add_argument("--label", default="default", type=str, help="label for checkpoint")
parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
parser.add_argument("--wd", default=5e-4, type=float, help="weight decay")
parser.add_argument("--min-lr", default=2e-4, type=float, help="minimal learning rate")
parser.add_argument("--dataset", type=str, default="cifar10", help="cifar10 or cifar100")
parser.add_argument("--b", type=int, default=128, help="batch size")
parser.add_argument("--resume", "-r", action="store_true", help="resume from checkpoint")
parser.add_argument(
    "--pretrained",
    action="store_true",
    default=False,
    help="Start with pretrained version of specified network (if avail)",
)
parser.add_argument("--num-classes", type=int, default=10, metavar="N", help="number of label classes (default: 1000)")
parser.add_argument("--model", default="T2t_vit_14", type=str, metavar="MODEL", help="Name of model to train")
parser.add_argument("--drop", type=float, default=0.0, metavar="PCT", help="Dropout rate (default: 0.0)")
parser.add_argument(
    "--drop-connect",
    type=float,
    default=None,
    metavar="PCT",
    help="Drop connect rate, DEPRECATED, use drop-path (default: None)",
)
parser.add_argument("--drop-path", type=float, default=0.1, metavar="PCT", help="Drop path rate (default: None)")
parser.add_argument("--drop-block", type=float, default=None, metavar="PCT", help="Drop block rate (default: None)")
parser.add_argument(
    "--gp",
    default=None,
    type=str,
    metavar="POOL",
    help="Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.",
)
parser.add_argument(
    "--img-size", type=int, default=224, metavar="N", help="Image patch size (default: None => model default)"
)
# CHANGED removed parameters not accepted by T2T_ViT
# parser.add_argument(
#     "--bn-tf",
#     action="store_true",
#     default=False,
#     help="Use Tensorflow BatchNorm defaults for models that support it (default: False)",
# )
# parser.add_argument("--bn-momentum", type=float, default=None, help="BatchNorm momentum override (if not None)")
# parser.add_argument("--bn-eps", type=float, default=None, help="BatchNorm epsilon override (if not None)")
parser.add_argument(
    "--initial-checkpoint",
    default="",
    type=str,
    metavar="PATH",
    help="Initialize model from this checkpoint (default: none)",
)
# Transfer learning
parser.add_argument("--transfer-learning", default=True, help="Enable transfer learning")
parser.add_argument("--transfer-model", type=str, default="saved_models/downloaded/imagenet/81.5_T2T_ViT_14.pth", 
                    help="Path to pretrained model for transfer learning")
parser.add_argument(
    "--transfer-ratio", type=float, default=0.01, help="lr ratio between classifier and backbone in transfer learning"
)

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0.0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print("==> Preparing data..")
transform_train = transforms.Compose(
    [
        transforms.Resize(args.img_size, transforms.InterpolationMode.BILINEAR),
        transforms.RandomRotation(10),  # CHANGED added random rotation
        transforms.RandomCrop(args.img_size, padding=(args.img_size // 8)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.Resize(args.img_size, transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

if args.dataset == "cifar10":
    args.num_classes = 10
    root = PROJECT_ROOT / "CIFAR_10_data"
    trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)

elif args.dataset == "cifar100":
    args.num_classes = 100
    trainset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)

elif args.dataset == "gastro":
    args.num_classes = 2
    train_set_full = GastroDataset(root = "./data")
    train_set_size = int(len(train_set_full) * 0.9)
    valid_set_size = len(train_set_full) - train_set_size
    trainset, testset = random_split(train_set_full, [train_set_size, valid_set_size])

    dataloader_kwargs: dict[str, Any] = dict(num_workers=0, pin_memory=False, collate_fn=collate_gastro_batch) #, prefetch_factor=1, persistent_workers=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.b, shuffle=True, **dataloader_kwargs)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, **dataloader_kwargs)

else:
    print("Please use cifar10, cifar100 or gastro dataset.")

if args.dataset != "gastro":
    dataloader_kwargs: dict[str, Any] = dict(num_workers=0, pin_memory=False, prefetch_factor=1, persistent_workers=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.b, shuffle=True, **dataloader_kwargs)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, **dataloader_kwargs)

print(f"learning rate:{args.lr}, weight decay: {args.wd}")
# create T2T-ViT Model
print("==> Building model..")
net = t2t_vit_14(pretrained=False)
# net = create_model(
#     args.model,
#     pretrained=args.pretrained,
#     num_classes=args.num_classes,
#     drop_rate=args.drop,
#     drop_connect_rate=args.drop_connect,
#     drop_path_rate=args.drop_path,
#     drop_block_rate=args.drop_block,
#     global_pool=args.gp,
#     # CHANGED removed parameters not accepted by T2T_ViT
#     # bn_tf=args.bn_tf,
#     # bn_momentum=args.bn_momentum,
#     # bn_eps=args.bn_eps,
#     checkpoint_path=args.initial_checkpoint,
#     img_size=args.img_size,
# )

if args.transfer_learning:
    print("transfer learning, load t2t-vit pretrained model")
    # CHANGED strict from False to True
    load_for_transfer_learning(net, args.transfer_model, use_ema=True, strict=False, num_classes=args.num_classes)

net = net.to(device)
if device == "cuda":
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print("==> Resuming from checkpoint..")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
    checkpoint = torch.load("./checkpoint/ckpt.pth")
    net.load_state_dict(checkpoint["net"])
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]

criterion = nn.CrossEntropyLoss()

# set optimizer
if args.transfer_learning:
    print("set different lr for the t2t module, backbone and classifier(head) of T2T-ViT")
    parameters = [
        {"params": net.module.tokens_to_token.parameters(), "lr": args.transfer_ratio * args.lr},
        {"params": net.module.blocks.parameters(), "lr": args.transfer_ratio * args.lr},
        {"params": net.module.head.parameters()},
    ]
else:
    parameters = net.parameters()


optimizer = optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=args.wd)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=args.min_lr, T_max=60)


# Training
def train(epoch):
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets, segmentation, bboxes) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(
            batch_idx,
            len(trainloader),
            "Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (train_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
        )


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, segmentation, bboxes) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(
                batch_idx,
                len(testloader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (test_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
            )

    # Save checkpoint.
    # Changed: path of checkpoint.
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / "transfer" / args.dataset / f"{args.model}__{args.label}"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    acc = 100.0 * correct / total
    if acc > best_acc:
        print("Saving..")
        # CHANGED don't complicate the state dict. Was:
        # state = {"net": net.state_dict(), "acc": acc, "epoch": epoch}
        torch.save(net.state_dict(), checkpoint_dir / f"epoch-{epoch}_acc-{acc}.pth")
        best_acc = acc


for epoch in range(start_epoch, start_epoch + 50):
    begin_time = time.time()
    train(epoch)
    test(epoch)
    scheduler.step()
    end_time = time.time()
    print(f"Epoch {epoch} took {end_time - begin_time:.2f} seconds in total.")
