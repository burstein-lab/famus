import pickle
import math
import time
from typing import Any
import os
import re

import numpy as np
import torch
from torch import optim
from torch.nn import TripletMarginLoss
from torch.utils.data import DataLoader
from torch.autograd import detect_anomaly, set_detect_anomaly
from tqdm import tqdm

from app import logger
from app.model import VariableNet
from app.dataset import IterableTripletDataset
from app.sdfloader import SDFloader
from app import get_cfg


def _train_model(
    model: VariableNet,
    sdfloader: SDFloader,
    triplet_loss_module: TripletMarginLoss,
    device: Any,
    checkpoint_dir_path: str,
    num_epochs=10,
    batch_size=32,
    save_checkpoints=True,
    evaluation=True,
    num_workers=4,
) -> VariableNet:
    """
    Train a model using a triplet loss function.
    :param model: The model to train.
    :param sdfloader: The sdfloader to use.
    :param triplet_loss_module: The triplet loss function.
    :param device: The device to use.
    :param num_epochs: The number of epochs to train.
    :param batch_size: The batch size.
    :param evaluation: Whether to evaluate the model every 10th batch.
    :return: The trained model.
    """
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    dataset = IterableTripletDataset(sdfloader)
    dataset.start = 0
    dataset.end = len(sdfloader.anchors) - 1
    total_batches = dataset.get_num_batches(batch_size)

    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        overall_start = dataset.start
        overall_end = dataset.end
        per_worker = int(
            math.ceil((overall_end - overall_start) / float(worker_info.num_workers))
        )
        worker_id = worker_info.id
        dataset.start = overall_start + worker_id * per_worker
        dataset.end = min(dataset.start + per_worker, overall_end)

    if device == "cpu":
        torch.set_num_threads(num_workers)
        num_workers = 0
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=True,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=True,
            worker_init_fn=worker_init_fn,
        )
    last_k_losses = []
    moving_avg_losses = []
    last_k_eval_losses = []
    moving_avg_eval_losses = []
    eval_round = False
    x_for_moving_avg = []
    x_for_moving_avg_eval = []
    x = 0

    logger.info("Setting detect_anomaly")
    set_detect_anomaly(True)

    for epoch_num in tqdm(range(num_epochs), ascii=True, desc="Epoch"):
        start_time = time.time()
        t_loss = 0
        batch_num = 1
        last_time = time.time()
        batch: tuple

        for batch in dataloader:
            x += 1
            if save_checkpoints and x % 10_000 == 0:
                torch.save(model, checkpoint_dir_path + str(x) + "_checkpoint.pt")
            if evaluation and batch_num % 10 == 0:
                eval_round = True
                logger.info("Evaluating model")
            else:
                eval_round = False
            batch = [
                batch[0].float(),
                batch[1].float(),
                batch[2].float(),
            ]
            if not device == "cpu":
                batch = (
                    batch[0].to(device),
                    batch[1].to(device),
                    batch[2].to(device),
                )

            if eval_round:
                model.eval()
                with torch.no_grad():
                    preds = model(batch)
                model.train()
            else:
                preds = model(batch)
            ancor_preds, positive_preds, negative_preds = (
                preds[0],
                preds[1],
                preds[2],
            )

            loss = triplet_loss_module(ancor_preds, positive_preds, negative_preds)

            if not eval_round:
                t_loss += loss.item()
                optimizer.zero_grad()
                logger.info("Loss: " + str(loss.item()))
                try:
                    loss.backward()
                except RuntimeError as e:
                    logger.info(
                        "Error in loss backward. Loss: "
                        + str(loss.item())
                        + " Error: "
                        + str(e)
                    )

                    batch_str = "Batch: "
                    for b in batch:
                        batch_str += str(b)
                        batch_str += "\n====\n"
                    grad_str = "Gradients: "
                    for p in model.parameters():
                        grad_str += str(p.grad)
                        grad_str += "\n====\n"
                    logger.info(batch_str)
                    logger.info(grad_str)

                    # exit(1)
                optimizer.step()
                last_k_losses.append(loss.item())
                if len(last_k_losses) >= 1000:
                    last_k_losses = last_k_losses[-1000:]
                    moving_avg_loss = np.mean(last_k_losses)
                    moving_avg_losses.append(moving_avg_loss)
                    x_for_moving_avg.append(x)
            else:
                last_k_eval_losses.append(loss.item())
                if len(last_k_eval_losses) >= 1000:
                    last_k_eval_losses = last_k_eval_losses[-1000:]
                    moving_avg_eval_loss = np.mean(last_k_eval_losses)
                    moving_avg_eval_losses.append(moving_avg_eval_loss)
                    x_for_moving_avg_eval.append(x)
            batch_time = time.time() - last_time

            msg = (
                "Epoch: "
                + str(epoch_num)
                + " Batch: "
                + str(batch_num)
                + "/"
                + str(total_batches)
                + " Batch time: "
                + str(batch_time)
                + " Loss: "
                + str(loss)
            )
            if len(last_k_losses) >= 1000:
                msg += " Moving average loss: " + str(moving_avg_loss)
            if len(last_k_eval_losses) >= 1000:
                msg += " Moving average eval loss: " + str(moving_avg_eval_loss)
            logger.info(msg)
            batch_num += 1
            last_time = time.time()

        epoch_time = time.time() - start_time
        logger.info("Epoch: " + str(epoch_num) + " Epoch time: " + str(epoch_time))
        epoch_num += 1

    return model


def load_latest_checkpoint(checkpoint_dir_path: str) -> VariableNet:
    files = os.listdir(checkpoint_dir_path)
    pattern = re.compile(r"\d+_checkpoint\.pt")
    files = [f for f in files if pattern.match(f)]
    if not files:
        return None
    files = sorted(files, key=lambda x: int(x.split("_")[0]))
    latest_checkpoint = files[-1]
    return torch.load(checkpoint_dir_path + latest_checkpoint)


def train(
    sdfloader_path: str,
    output_path: str,
    device: str,
    num_epochs=10,
    batch_size=32,
    save_checkpoints=False,
    checkpoint_dir_path=None,
    evaluation=True,
    num_workers=4,
) -> None:
    if save_checkpoints and not checkpoint_dir_path:
        raise ValueError(
            "Checkpoint dir path is required when save_checkpoints is True"
        )
    if checkpoint_dir_path:
        if not os.path.exists(checkpoint_dir_path):
            os.mkdir(checkpoint_dir_path)
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
            raise ValueError("Cuda is not available")
        device = torch.device("cuda")

    with open(sdfloader_path, "rb") as f:
        sdfloader: SDFloader = pickle.load(f)
    logger.info("sdfloader_path: " + sdfloader_path)
    logger.info("output_path: " + output_path)
    logger.info("device:" + str(device))
    input_size = sdfloader.sdf.matrix.shape[1]
    logger.info("input size: " + str(input_size))
    if not checkpoint_dir_path:
        snn_model = VariableNet(input_size, 2, 320).to(torch.float32)
        snn_model.to(device)
    else:
        result = load_latest_checkpoint(checkpoint_dir_path)
        if result is None:
            snn_model = VariableNet(input_size, 2, 320).to(torch.float32)
            snn_model.to(device)
        else:
            snn_model = result
    logger.info("Starting to train on device: " + str(device))
    loss = TripletMarginLoss(margin=1, p=2)
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
        num_workers=num_workers,
    )
    torch.save(model, output_path)
