import os
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Set

import numpy as np
from scipy.sparse import csr_matrix

from famus.logging import logger
from famus.model import MLP

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    logger.warning(
        "PyTorch is not installed. Please install PyTorch to use the training module."
    )
    from famus.model import nn  # imports dummy nn.Module class


class SDFDataset:
    def __init__(self, sdf):
        self.sparse_matrix: csr_matrix = sdf.matrix
        self.labels = sdf.labels
        self.index_ids = sdf.index_ids

        self.label_to_indices = defaultdict(list)
        self.unknown_indices = []
        self.valid_labels = []

        for idx, index_id in enumerate(self.index_ids):
            labels = self.labels[index_id]
            if len(labels) == 1 and labels[0] == "unknown":
                self.unknown_indices.append(idx)
            else:
                for label in labels:
                    self.label_to_indices[label].append(idx)

        # Keep only labels with at least 2 samples for positive pairs
        self.valid_labels = [
            label
            for label, indices in self.label_to_indices.items()
            if len(indices) >= 2
        ]

        if len(self.valid_labels) == 0:
            raise ValueError("No labels have at least 2 samples")

        if len(self.unknown_indices) == 0:
            logger.warning("No unknown samples available for negative sampling")

        # Log statistics
        label_sizes = [len(self.label_to_indices[label]) for label in self.valid_labels]
        logger.debug(f"Total samples: {len(self)}")
        logger.debug(f"Valid labels: {len(self.valid_labels)}")
        logger.debug(f"Unknown samples: {len(self.unknown_indices)}")
        logger.debug(
            f"Label sizes - min: {min(label_sizes)}, max: {max(label_sizes)}, "
            f"median: {np.median(label_sizes):.0f}"
        )

    def __len__(self):
        return self.sparse_matrix.shape[0]

    def get_sample(self, idx):
        """Get a single sample as dense tensor"""
        row = self.sparse_matrix[idx].toarray().flatten()
        return torch.tensor(row, dtype=torch.float32)

    def get_labels(self, idx) -> Set[str]:
        """Get labels for a sample"""
        return set(self.labels[self.index_ids[idx]])


def create_subset_dataset(parent_dataset: SDFDataset, subset_indices: List[int]):
    class SubsetSDFDataset:
        def __init__(self, parent, indices):
            self.parent = parent
            self.subset_indices = indices
            self.sparse_matrix = parent.sparse_matrix[indices]
            self.index_ids = [parent.index_ids[i] for i in indices]
            self.labels = parent.labels

            self.label_to_indices = defaultdict(list)
            self.unknown_indices = []
            self.valid_labels = []

            for new_idx, old_idx in enumerate(indices):
                labels = self.get_labels(new_idx)
                if len(labels) == 1 and "unknown" in labels:
                    self.unknown_indices.append(new_idx)
                else:
                    for label in labels:
                        self.label_to_indices[label].append(new_idx)

            self.valid_labels = [
                label for label, idxs in self.label_to_indices.items() if len(idxs) >= 2
            ]

        def get_sample(self, idx):
            old_idx = self.subset_indices[idx]
            return self.parent.get_sample(old_idx)

        def get_labels(self, idx) -> Set[str]:
            old_idx = self.subset_indices[idx]
            return self.parent.get_labels(old_idx)

        def __len__(self):
            return len(self.subset_indices)

    return SubsetSDFDataset(parent_dataset, subset_indices)


def create_train_val_split(
    dataset: SDFDataset,
    min_samples_for_val: int = 10,
    val_samples_per_label: int = 2,
    val_n_labels: int = 100,
    val_n_unknowns: int = 50,
    seed: int = 42,
):
    """
    Create train/val split that respects label support constraints.
    """
    random.seed(seed)
    np.random.seed(seed)

    eligible_labels = []

    label_to_single_label_indices = {}
    for label in dataset.valid_labels:
        label_to_single_label_indices[label] = [
            idx
            for idx in dataset.label_to_indices[label]
            if len(dataset.get_labels(idx)) == 1
        ]

    for label in dataset.valid_labels:
        n_samples = len(label_to_single_label_indices[label])

        if n_samples >= min_samples_for_val:
            eligible_labels.append(label)

    if len(eligible_labels) == 0:
        raise ValueError(f"No labels have at least {min_samples_for_val} samples")

    val_n_labels = min(val_n_labels, len(eligible_labels))

    val_labels = np.random.choice(eligible_labels, size=val_n_labels, replace=False)

    train_indices = []
    val_indices = []
    val_label_set = set(val_labels)

    for label in dataset.valid_labels:
        label_samples = label_to_single_label_indices[label]

        if label in val_label_set:
            shuffled = list(label_samples)
            random.shuffle(shuffled)
            n_val = min(val_samples_per_label, len(shuffled) - 4)
            val_indices.extend(shuffled[:n_val])
            train_indices.extend(shuffled[n_val:])
        else:
            train_indices.extend(label_samples)

    if len(dataset.unknown_indices) > 0:
        shuffled_unknowns = list(dataset.unknown_indices)
        random.shuffle(shuffled_unknowns)
        n_val_unknowns = min(val_n_unknowns, len(shuffled_unknowns) // 2)
        val_indices.extend(shuffled_unknowns[:n_val_unknowns])
        train_indices.extend(shuffled_unknowns[n_val_unknowns:])

    assert len(set(train_indices) & set(val_indices)) == 0

    logger.debug("Split created:")
    logger.debug(f"  Train: {len(train_indices)} samples")
    logger.debug(
        f"  Val: {len(val_indices)} samples ({len(val_indices) / (len(train_indices) + len(val_indices)) * 100:.1f}%)"
    )

    random.seed()
    np.random.seed()

    return train_indices, val_indices, list(val_labels)


class PKBatchSampler:
    def __init__(
        self,
        dataset: SDFDataset,
        p_labels: int = 8,
        k_per_label: int = 4,
        n_negatives: int = 16,
        num_batches: int = 100,
        label_sampling_strategy: str = "sqrt_inverse",
        temperature: float = 1.0,
    ):
        """
        Args:
            label_sampling_strategy:
                - "uniform": Equal probability for all labels
                - "inverse": Weight = 1/n_samples (strong balancing)
                - "sqrt_inverse": Weight = 1/sqrt(n_samples) (moderate, recommended)
                - "log_inverse": Weight = 1/log(n_samples+1) (weak balancing)
            temperature: Higher = more uniform, lower = more aggressive rebalancing
        """
        self.dataset = dataset
        self.p_labels = p_labels
        self.k_per_label = k_per_label
        self.n_negatives = n_negatives
        self.num_batches = num_batches
        self.label_sampling_strategy = label_sampling_strategy
        self.temperature = temperature

        # Compute label weights
        self.label_weights = self._compute_label_weights()
        self.label_probs = self.label_weights / self.label_weights.sum()

        self._log_sampling_stats()

    def _compute_label_weights(self):
        """Compute sampling weight for each label"""
        weights = []

        for label in self.dataset.valid_labels:
            n_samples = len(self.dataset.label_to_indices[label])

            if self.label_sampling_strategy == "uniform":
                weight = 1.0
            elif self.label_sampling_strategy == "inverse":
                weight = 1.0 / n_samples
            elif self.label_sampling_strategy == "sqrt_inverse":
                weight = 1.0 / np.sqrt(n_samples)
            elif self.label_sampling_strategy == "log_inverse":
                weight = 1.0 / np.log(n_samples + 1)
            else:
                raise ValueError(f"Unknown strategy: {self.label_sampling_strategy}")

            weight = weight ** (1.0 / self.temperature)
            weights.append(weight)

        return np.array(weights)

    def _log_sampling_stats(self):
        """Log statistics about label sampling distribution"""
        label_sizes = [
            len(self.dataset.label_to_indices[label])
            for label in self.dataset.valid_labels
        ]

        logger.info(
            f"Batch sampler - strategy: {self.label_sampling_strategy}, temp: {self.temperature}"
        )

        if self.label_sampling_strategy != "uniform":
            small_idx = np.argmin(label_sizes)
            large_idx = np.argmax(label_sizes)

            small_prob = self.label_probs[small_idx]
            large_prob = self.label_probs[large_idx]

            logger.debug(
                f"  Smallest label ({label_sizes[small_idx]} samples): {small_prob * 100:.3f}% probability"
            )
            logger.debug(
                f"  Largest label ({label_sizes[large_idx]} samples): {large_prob * 100:.3f}% probability"
            )
            logger.debug(f"  Rebalancing ratio: {small_prob / large_prob:.1f}x")

    def __iter__(self):
        for _ in range(self.num_batches):
            batch_indices = []
            batch_labels_used = set()

            sampled_labels = np.random.choice(
                self.dataset.valid_labels,
                size=min(self.p_labels, len(self.dataset.valid_labels)),
                replace=False,
            )

            # Sample K instances per label
            for label in sampled_labels:
                available_indices = self.dataset.label_to_indices[label]
                k = min(self.k_per_label, len(available_indices))
                sampled_indices = random.sample(available_indices, k)
                batch_indices.extend(sampled_indices)

                for idx in sampled_indices:
                    batch_labels_used.update(self.dataset.get_labels(idx))

            # Add negative samples
            negative_candidates = self.dataset.unknown_indices.copy()

            for label, indices in self.dataset.label_to_indices.items():
                if label not in batch_labels_used:
                    negative_candidates.extend(indices)

            negative_candidates = list(set(negative_candidates) - set(batch_indices))

            if len(negative_candidates) > 0:
                n = min(self.n_negatives, len(negative_candidates))
                negatives = random.sample(negative_candidates, n)
                batch_indices.extend(negatives)

            yield batch_indices

    def __len__(self):
        return self.num_batches


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, indices, dataset):
        batch_size = embeddings.size(0)
        device = embeddings.device

        label_matrix = torch.zeros(
            batch_size, batch_size, dtype=torch.bool, device=device
        )
        has_unknown = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for i, idx_i in enumerate(indices):
            labels_i = dataset.get_labels(idx_i)
            is_unknown_i = len(labels_i) == 1 and "unknown" in labels_i
            has_unknown[i] = is_unknown_i

            if not is_unknown_i:
                for j, idx_j in enumerate(indices):
                    if i != j:
                        labels_j = dataset.get_labels(idx_j)
                        is_unknown_j = len(labels_j) == 1 and "unknown" in labels_j

                        if (
                            not is_unknown_j
                            and len(labels_i.intersection(labels_j)) > 0
                        ):
                            label_matrix[i, j] = True

        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature
        mask_self = torch.eye(batch_size, dtype=torch.bool, device=device)
        similarity = similarity.masked_fill(mask_self, -float("inf"))
        losses = []

        for i in range(batch_size):
            if has_unknown[i]:
                continue
            positives_mask = label_matrix[i] & ~has_unknown
            if not positives_mask.any():
                continue
            negatives_mask = ~label_matrix[i] | has_unknown
            negatives_mask[i] = False
            if not negatives_mask.any():
                continue

            pos_sims = similarity[i][positives_mask]
            all_sims = similarity[i][positives_mask | negatives_mask]

            pos_log_sum = torch.logsumexp(pos_sims, dim=0)
            all_log_sum = torch.logsumexp(all_sims, dim=0)

            loss = -(pos_log_sum - all_log_sum)
            losses.append(loss)

        if len(losses) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return torch.stack(losses).mean()


class SupervisedContrastiveLossOriginal(nn.Module):
    """
    Original paper version that evaluates each positive separately.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, indices, dataset):
        batch_size = embeddings.size(0)
        device = embeddings.device

        # Build label matrix
        label_matrix = torch.zeros(
            batch_size, batch_size, dtype=torch.bool, device=device
        )
        has_unknown = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for i, idx_i in enumerate(indices):
            labels_i = dataset.get_labels(idx_i)
            is_unknown_i = len(labels_i) == 1 and "unknown" in labels_i
            has_unknown[i] = is_unknown_i

            if not is_unknown_i:
                for j, idx_j in enumerate(indices):
                    if i != j:
                        labels_j = dataset.get_labels(idx_j)
                        is_unknown_j = len(labels_j) == 1 and "unknown" in labels_j

                        if (
                            not is_unknown_j
                            and len(labels_i.intersection(labels_j)) > 0
                        ):
                            label_matrix[i, j] = True

        # Compute similarity
        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature
        mask_self = torch.eye(batch_size, dtype=torch.bool, device=device)
        similarity = similarity.masked_fill(mask_self, -float("inf"))

        losses = []

        for i in range(batch_size):
            if has_unknown[i]:
                continue

            positives_mask = label_matrix[i] & ~has_unknown
            if not positives_mask.any():
                continue

            negatives_mask = ~label_matrix[i] | has_unknown
            negatives_mask[i] = False

            if not negatives_mask.any():
                continue

            pos_sims = similarity[i][positives_mask]
            all_sims = similarity[i][positives_mask | negatives_mask]

            # Compute denominator once (sum over all)
            all_log_sum = torch.logsumexp(all_sims, dim=0)

            # Sum OUTSIDE the log - evaluate each positive separately
            for pos_sim in pos_sims:
                # Loss for this specific positive
                loss_per_positive = -(pos_sim - all_log_sum)
                losses.append(loss_per_positive)

        if len(losses) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Average over ALL positive pairs (not per anchor)
        return torch.stack(losses).mean()


def compute_distance_metrics(embeddings, indices, dataset, device):
    batch_size = embeddings.size(0)

    label_matrix = torch.zeros(batch_size, batch_size, dtype=torch.bool, device=device)
    has_unknown = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for i, idx_i in enumerate(indices):
        labels_i = dataset.get_labels(idx_i)
        is_unknown_i = len(labels_i) == 1 and "unknown" in labels_i
        has_unknown[i] = is_unknown_i

        if not is_unknown_i:
            for j, idx_j in enumerate(indices):
                if i != j:
                    labels_j = dataset.get_labels(idx_j)
                    is_unknown_j = len(labels_j) == 1 and "unknown" in labels_j

                    if not is_unknown_j and len(labels_i.intersection(labels_j)) > 0:
                        label_matrix[i, j] = True

    distances = torch.cdist(embeddings, embeddings, p=2)

    hardest_pos_dists = []
    hardest_neg_dists = []
    mean_pos_dists = []
    mean_neg_dists = []

    for i in range(batch_size):
        if has_unknown[i]:
            continue

        positives_mask = label_matrix[i] & ~has_unknown
        if not positives_mask.any():
            continue

        negatives_mask = (~label_matrix[i] | has_unknown) & (
            torch.arange(batch_size, device=device) != i
        )
        if not negatives_mask.any():
            continue

        pos_dists = distances[i][positives_mask]
        neg_dists = distances[i][negatives_mask]

        # Keep as tensors, not Python scalars
        hardest_pos_dists.append(pos_dists.max())
        hardest_neg_dists.append(neg_dists.min())
        mean_pos_dists.append(pos_dists.mean())
        mean_neg_dists.append(neg_dists.mean())

    if len(hardest_pos_dists) == 0:
        return None

    hardest_pos_tensor = torch.stack(hardest_pos_dists)
    hardest_neg_tensor = torch.stack(hardest_neg_dists)
    mean_pos_tensor = torch.stack(mean_pos_dists)
    mean_neg_tensor = torch.stack(mean_neg_dists)

    metrics = {
        "mean_hardest_pos_dist": hardest_pos_tensor.mean().item(),
        "mean_hardest_neg_dist": hardest_neg_tensor.mean().item(),
        "hardest_distance_ratio": (
            hardest_neg_tensor.mean() / (hardest_pos_tensor.mean() + 1e-8)
        ).item(),
        "mean_pos_dist": mean_pos_tensor.mean().item(),
        "mean_neg_dist": mean_neg_tensor.mean().item(),
        "mean_distance_ratio": (
            mean_neg_tensor.mean() / (mean_pos_tensor.mean() + 1e-8)
        ).item(),
    }

    return metrics


def create_fixed_eval_batch(
    dataset, p_labels=16, k_per_label=8, n_negatives=32, seed=42
):
    random.seed(seed)
    np.random.seed(seed)

    batch_indices = []
    batch_labels_used = set()

    sampled_labels = random.sample(
        dataset.valid_labels, min(p_labels, len(dataset.valid_labels))
    )

    for label in sampled_labels:
        available = dataset.label_to_indices[label]
        k = min(k_per_label, len(available))
        batch_indices.extend(random.sample(available, k))

        for idx in batch_indices[-k:]:
            batch_labels_used.update(dataset.get_labels(idx))

    negative_candidates = dataset.unknown_indices.copy()

    for label, indices in dataset.label_to_indices.items():
        if label not in batch_labels_used:
            negative_candidates.extend(indices[: min(10, len(indices))])

    negative_candidates = list(set(negative_candidates) - set(batch_indices))

    if len(negative_candidates) > 0:
        n = min(n_negatives, len(negative_candidates))
        batch_indices.extend(random.sample(negative_candidates, n))

    random.seed()
    np.random.seed()

    return batch_indices


def _train(
    train_dataset: SDFDataset,
    val_dataset: SDFDataset,
    input_dim: int,
    embedding_dim: int = 320,
    hidden_dims: Optional[List[int]] = None,
    temperature: float = 0.25,
    p_labels: int = 8,
    k_per_label: int = 4,
    n_negatives: int = 16,
    label_sampling_strategy: str = "sqrt_inverse",
    sampling_temperature: float = 1.0,
    num_epochs: int = 50,
    batches_per_epoch: int = 10_000,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.0,
    log_metrics_every: int = 100,
    val_eval_every: int = 2000,
    checkpoint_dir: str = "checkpoints",
    device: str = "cuda",
    use_wandb: bool = False,
    wandb_project: str = "embedding_training",
    overwrite_checkpoint: bool = False,
    continue_from_checkpoint: bool = False,
    wandb_api_key_path: str = "wandb_api_key.txt",
    return_best: bool = True,
):
    if (
        not overwrite_checkpoint
        and not continue_from_checkpoint
        and Path(f"{checkpoint_dir}/checkpoint_best.pt").exists()
    ):
        raise FileExistsError(
            f"Checkpoint directory {checkpoint_dir} already exists. "
            f"Use 'overwrite_checkpoint=True' to overwrite or 'continue_from_checkpoint=True' to continue training."
        )

    if continue_from_checkpoint and overwrite_checkpoint:
        raise ValueError(
            "Cannot use both 'overwrite_checkpoint' and 'continue_from_checkpoint' options simultaneously."
        )

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")
    if continue_from_checkpoint:
        checkpoint_files = list(Path(checkpoint_dir).glob("checkpoint_epoch_*.pt"))
        if len(checkpoint_files) == 0:
            raise FileNotFoundError(f"No checkpoint files found in '{checkpoint_dir}'.")
        latest_checkpoint = max(
            checkpoint_files, key=lambda x: int(x.stem.split("_")[-1])
        )

        model = MLP.load_from_state(str(latest_checkpoint), device=device)

        logger.info(f"Continuing training from checkpoint: {latest_checkpoint}")

    else:
        model = MLP(
            input_dim=input_dim, embedding_dim=embedding_dim, hidden_dims=hidden_dims
        )
        model.to(device)

    criterion = SupervisedContrastiveLoss(temperature=temperature)

    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    train_track_batch = create_fixed_eval_batch(train_dataset, seed=42)
    val_batch = create_fixed_eval_batch(val_dataset, seed=43)

    logger.debug("Fixed batches created:")
    logger.debug(f"  Train tracking: {len(train_track_batch)} samples")
    logger.debug(f"  Validation: {len(val_batch)} samples")

    if use_wandb:
        import wandb

        try:
            with open(wandb_api_key_path, "r") as f:
                wandb.login(key=f.read().strip())
        except:
            logger.warning("Could not load wandb API key, attempting anonymous login")

        wandb.init(
            project=wandb_project,
            config={
                "architecture": {
                    "embedding_dim": embedding_dim,
                    "hidden_dims": hidden_dims or [320, 320, 320],
                },
                "loss": {
                    "temperature": temperature,
                },
                "sampling": {
                    "p_labels": p_labels,
                    "k_per_label": k_per_label,
                    "n_negatives": n_negatives,
                    "strategy": label_sampling_strategy,
                    "temperature": sampling_temperature,
                },
                "optimization": {
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "num_epochs": num_epochs,
                    "batches_per_epoch": batches_per_epoch,
                },
                "data": {
                    "train_samples": len(train_dataset),
                    "val_samples": len(val_dataset),
                    "train_labels": len(train_dataset.valid_labels),
                    "val_labels": len(val_dataset.valid_labels),
                },
            },
        )

    logger.debug("=" * 80)
    logger.debug("Starting training with settings:")
    logger.debug(f"  Sampling: {label_sampling_strategy}")
    logger.debug(f"  Device: {device}")
    logger.debug("=" * 80)

    global_step = 0
    best_val_ratio = 0.0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        batch_sampler = PKBatchSampler(
            train_dataset,
            p_labels=p_labels,
            k_per_label=k_per_label,
            n_negatives=n_negatives,
            num_batches=batches_per_epoch,
            label_sampling_strategy=label_sampling_strategy,
            temperature=sampling_temperature,
        )

        for batch_indices in batch_sampler:
            batch_data = torch.stack(
                [train_dataset.get_sample(i) for i in batch_indices]
            ).to(device)
            embeddings = model(batch_data)
            loss = criterion(embeddings, batch_indices, train_dataset)

            if global_step % log_metrics_every == 0:
                model.eval()
                with torch.no_grad():
                    track_data = torch.stack(
                        [train_dataset.get_sample(i) for i in train_track_batch]
                    ).to(device)
                    track_embeddings = model(track_data)
                    track_metrics = compute_distance_metrics(
                        track_embeddings, train_track_batch, train_dataset, device
                    )

                    if track_metrics is not None:
                        # Log all metrics to wandb (including current loss)
                        if use_wandb:
                            wandb_metrics = {
                                f"train_track_{k}": v for k, v in track_metrics.items()
                            }
                            # Also log the current training loss at this checkpoint
                            wandb_metrics["train_track_loss"] = loss.item()
                            wandb.log(wandb_metrics, step=global_step)

                        logger.debug(
                            f"Step {global_step}: [TRAIN] Loss={loss.item():.4f}, "
                            f"Hard ratio={track_metrics['hardest_distance_ratio']:.3f}, "
                            f"Pos={track_metrics['mean_hardest_pos_dist']:.3f}, "
                            f"Neg={track_metrics['mean_hardest_neg_dist']:.3f}"
                        )
                model.train()

            # True validation metrics (less frequent)
            if global_step % val_eval_every == 0 and global_step > 0:
                model.eval()
                with torch.no_grad():
                    val_data = torch.stack(
                        [val_dataset.get_sample(i) for i in val_batch]
                    ).to(device)
                    val_embeddings = model(val_data)
                    val_metrics = compute_distance_metrics(
                        val_embeddings, val_batch, val_dataset, device
                    )

                    if val_metrics is not None:
                        # Log all validation metrics to wandb
                        if use_wandb:
                            wandb_val_metrics = {
                                f"val_{k}": v for k, v in val_metrics.items()
                            }
                            wandb.log(wandb_val_metrics, step=global_step)

                        logger.info(
                            f"Step {global_step}: [VAL] "
                            f"Hard ratio={val_metrics['hardest_distance_ratio']:.3f}, "
                            f"Mean ratio={val_metrics['mean_distance_ratio']:.3f}, "
                            f"Pos={val_metrics['mean_hardest_pos_dist']:.3f}, "
                            f"Neg={val_metrics['mean_hardest_neg_dist']:.3f}"
                        )

                        if val_metrics["hardest_distance_ratio"] > best_val_ratio:
                            best_val_ratio = val_metrics["hardest_distance_ratio"]

                            torch.save(
                                {
                                    "step": int(global_step),
                                    "epoch": int(epoch),
                                    "model_state_dict": model.state_dict(),
                                    "optimizer_state_dict": optimizer.state_dict(),
                                    "val_ratio": float(best_val_ratio),
                                },
                                best_path,
                            )
                            logger.info(
                                f"New best validation ratio: {best_val_ratio:.3f}"
                            )

                model.train()

            # Optimization step
            if loss.item() > 0:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1

            global_step += 1

        scheduler.step()

        # Epoch summary
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} complete: "
                f"Avg Loss={avg_loss:.4f}, LR={scheduler.get_last_lr()[0]:.2e}"
            )

        # Save checkpoint
        checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch + 1}.pt"
        torch.save(
            {
                "epoch": int(epoch),
                "step": int(global_step),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": float(avg_loss) if num_batches > 0 else None,
            },
            checkpoint_path,
        )

    if use_wandb:
        wandb.finish()

    logger.info("Training complete!")

    logger.info(f"Best validation ratio: {best_val_ratio:.3f}")

    if return_best:
        logger.info("Loading best model from checkpoint...")
        best_checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(best_checkpoint["model_state_dict"])

    return model


def train(
    sdf_path: str,
    checkpoint_dir: str,
    model_output_path: str,
    seed: int = 42,
    device: str = "cuda",
    embedding_dim: int = 320,
    hidden_dims: Optional[List[int]] = None,
    temperature: float = 0.25,
    p_labels: int = 16,
    k_per_label: int = 8,
    n_negatives: int = 64,
    label_sampling_strategy: str = "uniform",
    sampling_temperature: float = 1.0,
    num_epochs: int = 50,
    batches_per_epoch: int = 10_000,
    learning_rate: float = 1e-5,
    log_metrics_every: int = 100,
    val_eval_every: int = 2000,
    use_wandb: bool = True,
    wandb_project: str = "famus",
    wandb_api_key_path: str = "wandb_api_key.txt",
    overwrite_checkpoint: bool = False,
    continue_from_checkpoint: bool = False,
    return_best: bool = True,
):
    logger.info("Loading SDF...")
    sdf = pickle.load(open(sdf_path, "rb"))

    logger.info("Creating full dataset...")
    full_dataset = SDFDataset(sdf)

    logger.info("Creating train/val split...")
    train_indices, val_indices, val_labels = create_train_val_split(
        full_dataset,
        min_samples_for_val=15,
        val_samples_per_label=3,
        val_n_labels=100,
        val_n_unknowns=50,
        seed=seed,
    )

    train_dataset = create_subset_dataset(full_dataset, train_indices)
    val_dataset = create_subset_dataset(full_dataset, val_indices)

    logger.info("Datasets ready:")
    logger.info(
        f"  Train: {len(train_dataset)} samples, {len(train_dataset.valid_labels)} labels"
    )
    logger.info(
        f"  Val: {len(val_dataset)} samples, {len(val_dataset.valid_labels)} labels"
    )
    if hidden_dims is None:
        hidden_dims = [320, 320, 320]
    # Train model
    model = _train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        input_dim=train_dataset.sparse_matrix.shape[1],
        embedding_dim=embedding_dim,
        hidden_dims=hidden_dims,
        temperature=temperature,
        p_labels=p_labels,
        k_per_label=k_per_label,
        n_negatives=n_negatives,
        label_sampling_strategy=label_sampling_strategy,
        sampling_temperature=sampling_temperature,
        num_epochs=num_epochs,
        batches_per_epoch=batches_per_epoch,
        learning_rate=learning_rate,
        log_metrics_every=log_metrics_every,
        val_eval_every=val_eval_every,
        checkpoint_dir=checkpoint_dir,
        device=device,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_api_key_path=wandb_api_key_path,
        overwrite_checkpoint=overwrite_checkpoint,
        continue_from_checkpoint=continue_from_checkpoint,
        return_best=return_best,
    )

    logger.info("All done!")
    logger.info(f"Saving final model to {model_output_path}...")
    torch.save({"model_state_dict": model.state_dict()}, model_output_path)
    logger.info("Model saved.")
