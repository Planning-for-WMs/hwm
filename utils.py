import numpy as np
import torch
from pathlib import Path
from stable_pretraining import data as dt
from stable_pretraining.callbacks import RankMe
from stable_pretraining.callbacks.queue import find_or_create_queue_callback
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()

    def norm_fn(x):
        return ((x - mean) / std).float()

    normalizer = dt.transforms.WrapTorchTransform(norm_fn, source=source, target=target)
    return normalizer

class ModelObjectCallBack(Callback):
    """Callback to pickle model object after each epoch."""

    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        output_path = (
            self.dirpath
            / f"{self.filename}_epoch_{trainer.current_epoch + 1}_object.ckpt"
        )

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._dump_model(pl_module.model, output_path)

            # save final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._dump_model(pl_module.model, output_path)

    def _dump_model(self, model, path):
        try:
            torch.save(model, path)
        except Exception as e:
            print(f"Error saving model object: {e}")


class IntervalRankMe(RankMe):
    """RankMe gated by `every_n_epochs`."""
    def __init__(self, *args, every_n_epochs: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.every_n_epochs = every_n_epochs

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch,
                                batch_idx, dataloader_idx=0):
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        super().on_validation_batch_end(trainer, pl_module, outputs, batch,
                                        batch_idx, dataloader_idx)


class PCALatentViz(Callback):
    """Every `every_n_epochs` validations, run sklearn PCA(n=2) on a queue of
    embeddings and log a scatter plot to wandb."""

    def __init__(self, name: str = "hl_pca", target: str = "hl_emb",
                 queue_length: int = 4096, target_shape: int = 96,
                 every_n_epochs: int = 1):
        super().__init__()
        self.name = name
        self.target = target
        self.queue_length = queue_length
        self.target_shape = target_shape
        self.every_n_epochs = every_n_epochs
        self._queue = None

    def setup(self, trainer, pl_module, stage):
        if self._queue is None:
            self._queue = find_or_create_queue_callback(
                trainer, self.target, self.queue_length, self.target_shape,
                torch.float32, gather_distributed=True, create_if_missing=True,
            )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch,
                                batch_idx, dataloader_idx=0):
        if batch_idx > 0 or trainer.global_rank != 0:
            return
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        emb = self._queue.data
        if emb is None or emb.numel() == 0:
            return

        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt

        x = emb.detach().cpu().float().numpy()
        pca = PCA(n_components=2).fit(x)
        proj = pca.transform(x)
        ev = pca.explained_variance_ratio_

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(proj[:, 0], proj[:, 1], s=4, alpha=0.5)
        ax.set_title(f"{self.name} epoch {trainer.current_epoch + 1}  "
                     f"EVR=[{ev[0]:.2f}, {ev[1]:.2f}]")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

        if isinstance(trainer.logger, WandbLogger):
            import wandb
            trainer.logger.experiment.log({
                self.name: wandb.Image(fig),
                f"{self.name}/evr_pc1": float(ev[0]),
                f"{self.name}/evr_pc2": float(ev[1]),
            }, step=trainer.global_step)
        plt.close(fig)


class LatentViz3D(Callback):
    """3D scatter of a low-d (target_shape=3) latent queue — no PCA, plot the
    raw dimensions directly. Use for visualizing the MAE macro space when
    macro_action_dim=3. Samples come from whatever populates the queue, which
    at training time is the model's outputs on expert trajectories."""

    def __init__(self, name: str = "mae_3d", target: str = "mae_emb",
                 queue_length: int = 4096, target_shape: int = 3,
                 every_n_epochs: int = 1):
        super().__init__()
        self.name = name
        self.target = target
        self.queue_length = queue_length
        self.target_shape = target_shape
        self.every_n_epochs = every_n_epochs
        self._queue = None

    def setup(self, trainer, pl_module, stage):
        if self._queue is None:
            self._queue = find_or_create_queue_callback(
                trainer, self.target, self.queue_length, self.target_shape,
                torch.float32, gather_distributed=True, create_if_missing=True,
            )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch,
                                batch_idx, dataloader_idx=0):
        if batch_idx > 0 or trainer.global_rank != 0:
            return
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        emb = self._queue.data
        if emb is None or emb.numel() == 0:
            return

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

        x = emb.detach().cpu().float().numpy()
        assert x.shape[1] == 3, f"LatentViz3D expects 3-d target, got {x.shape[1]}"

        per_dim_std = x.std(axis=0)
        norms = np.linalg.norm(x, axis=1)

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], s=3, alpha=0.4, c=norms, cmap="viridis")
        ax.set_xlabel("d0"); ax.set_ylabel("d1"); ax.set_zlabel("d2")
        ax.set_title(f"{self.name}  epoch {trainer.current_epoch + 1}  "
                     f"std=[{per_dim_std[0]:.2f}, {per_dim_std[1]:.2f}, {per_dim_std[2]:.2f}]")

        if isinstance(trainer.logger, WandbLogger):
            import wandb
            trainer.logger.experiment.log({
                self.name: wandb.Image(fig),
                f"{self.name}/std_d0": float(per_dim_std[0]),
                f"{self.name}/std_d1": float(per_dim_std[1]),
                f"{self.name}/std_d2": float(per_dim_std[2]),
                f"{self.name}/norm_mean": float(norms.mean()),
            }, step=trainer.global_step)
        plt.close(fig)