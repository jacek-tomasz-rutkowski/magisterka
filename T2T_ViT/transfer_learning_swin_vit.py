# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.

"""Transfer pretrained Swin transformer to downstream dataset: CIFAR10/CIFAR100."""
import argparse
import time

from transformers import SwinForImageClassification, ViTForImageClassification
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from utils import progress_bar

from typing import Any
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

parser = argparse.ArgumentParser(description="PyTorch CIFAR10/CIFAR100 Training")
parser.add_argument("--label", default="default", type=str, help="label for checkpoint")
parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
parser.add_argument("--wd", default=5e-4, type=float, help="weight decay")
parser.add_argument("--min-lr", default=2e-4, type=float, help="minimal learning rate")
parser.add_argument("--b", type=int, default=128, help="batch size")
parser.add_argument("--dataset", type=str, default="cifar10", help="cifar10 or cifar100")
parser.add_argument("--model_name", type=str, default="swin", help="swin or vit")

parser.add_argument("--num-classes", type=int, default=10, metavar="N", help="number of label classes (default: 1000)")
parser.add_argument(
    "--img-size", type=int, default=224, metavar="N", help="Image patch size (default: None => model default)"
)
parser.add_argument(
    "--transfer-ratio", type=float, default=0.01, help="lr ratio between classifier and backbone in transfer learning"
)

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0.0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

def get_all_parent_layers(net, type):
    layers = []

    for name, l in net.named_modules():
        if isinstance(l, type):
            tokens = name.strip().split('.')

            layer = net
            for t in tokens[:-1]:
                if not t.isnumeric():
                    layer = getattr(layer, t)
                else:
                    layer = layer[int(t)]

            layers.append([layer, tokens[-1]])

    return layers

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

args.num_classes = 10
root = PROJECT_ROOT / "CIFAR_10_data"
trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)

dataloader_kwargs: dict[str, Any] = dict(num_workers=0, pin_memory=False) #, prefetch_factor=2, persistent_workers=False)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.b, shuffle=True, **dataloader_kwargs)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, **dataloader_kwargs)

if args.model_name == 'swin':
    net = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224").to(device)
    net.classifier = nn.Linear(net.classifier.in_features, args.num_classes).to(device)
    parameters = [
        {"params": net.swin.parameters(), "lr": args.transfer_ratio * args.lr},
        {"params": net.classifier.parameters()},
    ]

else:
    net = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224").to(device)
    net.classifier = nn.Linear(net.classifier.in_features, args.num_classes).to(device)
    parameters = [
        {"params": net.vit.parameters(), "lr": args.transfer_ratio * args.lr},
        {"params": net.classifier.parameters()},
    ]
    # we need to change lr for transfer learning



criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=args.wd)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=args.min_lr, T_max=60)


# Training
def train(epoch):
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        # x = inputs
        # for i in range(len(net.vit.encoder.layer)):
        #     x = net.vit.encoder.layer[i](x)
        #     print(f'After {net.vit.encoder.layer} values are {x}')
        outputs = net(inputs).logits
        print(outputs)

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
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs).logits
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
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / "transfer" / args.dataset / args.label
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    acc = 100.0 * correct / total
    if acc > best_acc:
        print("Saving..")
        torch.save(net.state_dict(), checkpoint_dir / f"{args.model_name}_epoch-{epoch}_acc-{acc}.pth")
        best_acc = acc


for epoch in range(start_epoch, start_epoch + 1):
    begin_time = time.time()
    train(epoch)
    test(epoch)
    scheduler.step()
    end_time = time.time()
    print(f"Epoch {epoch} took {end_time - begin_time:.2f} seconds in total.")
