import pickle
from typing import Any
import os
import re
from typing import Tuple

from famus.utils import now

try:
    import torch
    from torch import optim
    from torch.nn import TripletMarginLoss
except ImportError:
    from famus.logging import logger

    logger.warning(
        "PyTorch is not installed. Please install PyTorch to use the train module."
    )

from famus.logging import logger
from famus.model import MLP, load_from_state
from famus.sdfloader import SDFloader


def save_state(model: MLP, path: str, batch_num: int, epoch_num: int) -> None:
    epoch_dir = os.path.join(path, str(f"epoch_{epoch_num}"))
    os.makedirs(epoch_dir, exist_ok=True)
    save_path = os.path.join(epoch_dir, f"{batch_num}_checkpoint.pt")
    model.save_state(save_path)
    logger.info(f"Saved state to {save_path}")


def _train_model(
    model: MLP,
    sdfloader: SDFloader,
    triplet_loss_module,
    device: Any,
    checkpoint_dir_path: str,
    num_epochs=10,
    batch_size=32,
    save_checkpoints=True,
    save_every=100_000,
    evaluation=True,
    lr=0.001,
    start_epoch=0,
    start_batch=0,
    log_to_wandb=False,
) -> MLP:
    """
    Train a model using a triplet loss function.
    """
    if not isinstance(num_epochs, int):
        raise ValueError("num_epochs must be an integer")
    if not num_epochs > 0:
        raise ValueError("num_epochs must be greater than 0")
    if not isinstance(batch_size, int):
        raise ValueError("batch_size must be an integer")
    if not batch_size > 0:
        raise ValueError("batch_size must be greater than 0")
    if not isinstance(save_checkpoints, bool):
        raise ValueError("save_checkpoints must be a boolean")
    if not isinstance(evaluation, bool):
        raise ValueError("evaluation must be a boolean")
    if not isinstance(save_every, int):
        raise ValueError("save_every must be an integer")
    if not save_every > 0:
        raise ValueError("save_every must be greater than 0")
    if not isinstance(lr, float):
        raise ValueError("lr must be a float")
    if not lr > 0:
        raise ValueError("lr must be greater than 0")
    if not isinstance(start_epoch, int):
        raise ValueError("start_epoch must be an integer")
    if not start_epoch >= 0:
        raise ValueError("start_epoch must be greater than or equal to 0")
    if not isinstance(start_batch, int):
        raise ValueError("start_batch must be an integer")
    if not start_batch >= 0:
        raise ValueError("start_batch must be greater than or equal to 0")

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.train()
    total_batches = sdfloader.get_num_batches(batch_size)
    eval_round = False
    steps_taken = 1
    if log_to_wandb:
        import wandb

    for epoch_num in range(start_epoch, num_epochs):
        batch_num = 0
        batch: tuple
        for batch in sdfloader.triplet_batch_generator(batch_size=batch_size):
            batch = (
                batch[0].float().to(device),
                batch[1].float().to(device),
                batch[2].float().to(device),
            )

            if batch_num < start_batch and epoch_num <= start_epoch:
                logger.info("Skipping completed batches..")

                batch_num += 1
                continue
            # calculate euclidean distance
            # between the ancor and positive samples
            # and between the ancor and negative samples

            if batch_num == start_batch and epoch_num == start_epoch:
                logger.info(
                    f"Starting training from checkpoint epoch {epoch_num} and batch {batch_num}"
                )
            if save_checkpoints and steps_taken % save_every == 0:
                save_state(model, checkpoint_dir_path, batch_num, epoch_num)
            if (
                evaluation and batch_num % 10 == 0
            ):  # the dataset isn't shuffled after each epoch and batch size is static, so it's unnecessary to separate the evaluation set from the training set
                eval_round = True
            else:
                eval_round = False
            if not device == "cpu":
                batch = (
                    batch[0].float().to(device),
                    batch[1].float().to(device),
                    batch[2].float().to(device),
                )
            else:
                batch = (batch[0].float(), batch[1].float(), batch[2].float())
            if eval_round:
                model.eval()
                with torch.no_grad():
                    preds = model(batch)
                model.train()
            else:
                preds = model(batch)
            ancor_preds, positive_preds, negative_preds = preds[0], preds[1], preds[2]
            loss = triplet_loss_module(ancor_preds, positive_preds, negative_preds)
            loss_float = round(loss.item(), 4)
            if not eval_round:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            else:
                if log_to_wandb:
                    wandb.log(
                        {
                            "eval_loss": loss_float,
                            "epoch": epoch_num,
                            "batch_num": batch_num,
                            "steps_taken": steps_taken,
                        },
                        step=steps_taken,
                    )

            msg = (
                "Epoch: "
                + str(epoch_num)
                + " Batch: "
                + str(batch_num)
                + "/"
                + str(total_batches)
                + " Loss: "
                + str(loss_float)
            )
            if eval_round:
                msg += " (eval round)"

            logger.debug(msg)
            batch_num += 1
            steps_taken += 1
    return model


def get_latest_checkpoint(checkpoint_dir_path: str) -> Tuple[int, int]:
    if not os.path.exists(checkpoint_dir_path):
        raise ValueError(f"Path {checkpoint_dir_path} does not exist")
    epochs = os.listdir(checkpoint_dir_path)
    if not epochs:
        return None
    pattern = re.compile(r"epoch_\d+")
    filtered = [e for e in epochs if pattern.match(e)]
    latest_epoch = max([int(e.split("_")[1]) for e in filtered])
    epoch_dir = os.path.join(checkpoint_dir_path, f"epoch_{latest_epoch}")
    files = os.listdir(epoch_dir)
    pattern = re.compile(r"\d+_checkpoint\.pt")
    files = [f for f in files if pattern.match(f)]
    if not files:
        return None
    latest_checkpoint = max([int(f.split("_")[0]) for f in files])
    return latest_epoch, latest_checkpoint


def get_latest_checkpoint_path(checkpoint_dir_path: str) -> str:
    if latest_epoch_and_checkpoint := get_latest_checkpoint(checkpoint_dir_path):
        return (
            os.path.join(
                checkpoint_dir_path,
                f"epoch_{latest_epoch_and_checkpoint[0]}/{latest_epoch_and_checkpoint[1]}_checkpoint.pt",
            ),
            latest_epoch_and_checkpoint[0],
            latest_epoch_and_checkpoint[1],
        )
    return None


def train(
    sdfloader_path: str,
    output_path: str,
    device: str,
    num_epochs=10,
    batch_size=32,
    save_checkpoints=False,
    checkpoint_dir_path=None,
    evaluation=True,
    save_every=100_000,
    lr=0.001,
    n_processes=1,
    log_to_wandb=False,
    wandb_project="famus",
    wandb_api_key_path=None,
) -> None:
    if os.path.exists(output_path):
        logger.info("Output path already exists. Exiting.")
        return
    if save_checkpoints and not checkpoint_dir_path:
        raise ValueError(
            "Checkpoint dir path is required when save_checkpoints is True"
        )
    if checkpoint_dir_path:
        os.makedirs(checkpoint_dir_path, exist_ok=True)
    output_dir_path = os.path.dirname(output_path)
    if not os.path.exists(output_dir_path):
        raise ValueError(f"Path {output_dir_path} does not exist")
    if not isinstance(num_epochs, int):
        raise ValueError("num_epochs must be an integer")
    if not num_epochs > 0:
        raise ValueError("num_epochs must be greater than 0")
    if not isinstance(batch_size, int):
        raise ValueError("batch_size must be an integer")
    if not batch_size > 0:
        raise ValueError("batch_size must be greater than 0")

    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("cuda is not available in this environment")
        device = torch.device("cuda")
    elif device == "cpu":
        device = torch.device("cpu")
        torch.set_num_threads(n_processes)
    if log_to_wandb:
        import wandb

    with open(sdfloader_path, "rb") as f:
        sdfloader: SDFloader = pickle.load(f)
    logger.info("sdfloader_path: " + sdfloader_path)
    logger.info("output_path: " + output_path)
    logger.info("device:" + str(device))
    input_size = sdfloader.sdf.matrix.shape[1]
    logger.info("input size: " + str(input_size))
    logger.info(f"Learning rate: {lr}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Number of epochs: {num_epochs}")
    if checkpoint_dir_path:
        if checkpoint_data := get_latest_checkpoint_path(checkpoint_dir_path):
            state_path, latest_epoch, latest_batch = (
                checkpoint_data[0],
                checkpoint_data[1],
                checkpoint_data[2],
            )
            snn_model = load_from_state(state_path)
            logger.info(f"Loaded model from {state_path}")
        else:
            snn_model = MLP(input_size=input_size, num_layers=3, embedding_size=320)
            snn_model.to(device)
            latest_epoch = 0
            latest_batch = 0
    else:
        snn_model = MLP(input_size=input_size, num_layers=3, embedding_size=320)
        snn_model.to(device)
        latest_epoch = 0
        latest_batch = 0

    logger.info("Starting to train on device: " + str(device))
    loss = TripletMarginLoss(margin=1, p=2)
    if log_to_wandb:
        if wandb_api_key_path:
            with open(wandb_api_key_path, "r") as f:
                wandb.login(key=f.read().strip())
        else:
            wandb.login()
        time = now()
        model_dir = os.path.basename(os.path.dirname(output_path))
        run_name = f"{model_dir}_{time}"
        wandb.init(
            project=wandb_project,
            name=run_name,
            config={
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "device": str(device),
            },
        )
        wandb.watch(snn_model, log="all")
    else:
        logger.info("wandb logging is disabled.")
    if save_checkpoints:
        if not checkpoint_dir_path:
            raise ValueError(
                "checkpoint_dir_path is required when save_checkpoints is True"
            )
        if not os.path.exists(checkpoint_dir_path):
            os.makedirs(checkpoint_dir_path, exist_ok=True)
        logger.info(f"Checkpoints will be saved to {checkpoint_dir_path}")
    else:
        logger.info(
            "Checkpoints will not be saved. Set save_checkpoints=True to enable it."
        )
    if latest_epoch > 0 or latest_batch > 0:
        logger.info(
            f"Resuming training from epoch {latest_epoch} and batch {latest_batch}. "
            "If this is not intended, please delete the checkpoint directory."
        )
    else:
        logger.info("Starting training from scratch.")

    logger.info("Training model")
    model = _train_model(
        model=snn_model,
        sdfloader=sdfloader,
        triplet_loss_module=loss,
        device=device,
        num_epochs=num_epochs,
        batch_size=batch_size,
        evaluation=evaluation,
        save_checkpoints=save_checkpoints,
        checkpoint_dir_path=checkpoint_dir_path,
        start_epoch=latest_epoch,
        start_batch=latest_batch,
        save_every=save_every,
        lr=lr,
    )
    model.save_state(output_path)
    logger.info("Training complete")
    logger.info("Model saved to " + output_path)
