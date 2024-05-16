import time
import torch
import scipy.special
from tqdm import tqdm

from datasets.CIFAR_10_Dataset import PROJECT_ROOT, CIFAR_10_Datamodule, apply_masks
from vit_shapley.modules.surrogate import Surrogate


class BruteShap:
    """
    Computes shap values by brute force, from the definition.

    Args:
    - model: a model that takes masked images and outputs classification logits.
    - num_classes: int, the number of classes in the classification task.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        num_classes: int = 10,
    ):
        self.model = model
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_all_masks(K: int) -> torch.Tensor:
        """
        Return all 0-1 masks as a tensor of shape (2**K, K), dtype bool.

        Args:
        - K is num_players or num_players - 1.
        """
        if K > 63:
            raise ValueError("K should be at most 63")
        elif K > 31:
            t = torch.int64
        else:
            t = torch.int32
        bitmasks = torch.arange(2**K, dtype=t)
        which_bit = 2 ** torch.arange(K, dtype=t)
        masks = torch.bitwise_and(bitmasks.unsqueeze(-1), which_bit.unsqueeze(0))
        return masks.bool()

    def compute_model_on_masks(self, images: torch.Tensor, masks: torch.Tensor, compute_batch: int) -> torch.Tensor:
        """
        Compute values of the model for every image and every mask, return shape shape (N, 2**num_players, num_classes).

        Args:
        - images: shape (N, C, H, W), uint8 RGB.
        - masks: shape (2**num_players, num_players).
        - compute_batch: max number of masked images that will be passed to the model in one batch.
        """
        num_images, C, H, W = images.shape
        num_masks, num_players = masks.shape
        assert C == 3
        assert num_masks == 2 ** num_players

        num_masks_in_batch = compute_batch // num_images
        num_batches = (num_masks + num_masks_in_batch - 1) // num_masks_in_batch

        all_values = []
        self.model = self.model.to(self.device).eval()

        images = images.to(self.device)
        masks = masks.to(self.device)

        with torch.no_grad():
            for i in tqdm(range(num_batches), desc="Compute model values"):
                masks_slice = masks[i * num_masks_in_batch : min((i + 1) * num_masks_in_batch, num_masks)]
                # (num_masks_in_batch, num_players) or (num_masks % num_masks_in_batch, num_players)

                masked_images = apply_masks(images=images, masks=masks_slice.expand(num_images, -1, -1))
                # (num_images * len(masks_slice), C, H, W)

                values = self.model(masked_images).view(num_images, -1, self.num_classes).cpu()
                # (num_images, len(masks_slice), self.num_classes)

                all_values.append(values)

        return torch.cat(all_values, dim=1)

    def shap_values(self, images: torch.Tensor, num_players: int, compute_batch: int = 256) -> torch.Tensor:
        """
        Compute shapley values for the given images, return shape (N, num_players, num_classes).

        Args:
        - images: NCHW, uint8 RGB (where N is the number of images).
        - num_players: int, the number of patches that can be masked or not.
        - compute_batch: max number of masked images that will be passed to the model in one batch.
        """
        num_images = images.shape[0]

        all_masks = self.get_all_masks(K=num_players)
        # (2**num_players, num_players)
        almost_all_masks = self.get_all_masks(num_players - 1)
        # (2**(num_players - 1), num_players - 1)
        mask_size = torch.sum(almost_all_masks.long(), dim=-1)
        # (2**(num_players - 1),)
        coeffs = torch.Tensor([
            scipy.special.binom(num_players - 1, mask_size[i].item())
            for i in range(2 ** (num_players - 1))
        ]).float()
        # (2**(num_players - 1),)

        shap_vs = torch.zeros(num_images, num_players, self.num_classes)

        model_values = self.compute_model_on_masks(images, all_masks, compute_batch=compute_batch)
        # (B, 2**num_players, num_classes)

        for player in tqdm(range(num_players), desc="Compute shapley values"):
            includes_player = all_masks[:, player]
            # (2**num_players,)
            excludes_player = ~includes_player
            # (2**num_players,)

            diffs = model_values[:, includes_player, :] - model_values[:, excludes_player, :]
            # (B, 2**(num_players - 1), num_classes)

            shap_vs[:, player, :] = (diffs / coeffs.view(1, -1, 1)).sum(dim=1)

        return shap_vs / num_players


def main() -> None:
    surrogate = Surrogate.load_from_checkpoint(
        PROJECT_ROOT / "saved_models/surrogate/cifar10/v2/player16/t2t_vit.ckpt", map_location="cuda", strict=True
    )

    datamodule = CIFAR_10_Datamodule(
        num_players=16, num_mask_samples=1, paired_mask_samples=False, batch_size=3, num_workers=0
    )
    datamodule.setup()
    data = next(iter(datamodule.test_dataloader()))
    images, _ = data["images"], data["labels"]

    start_time = time.time()

    brute_shap = BruteShap(surrogate, num_classes=10)
    print(brute_shap.shap_values(images, num_players=16, compute_batch=256))

    print(f"Time of computation is {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
