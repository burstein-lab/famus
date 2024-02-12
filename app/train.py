import pickle
import time
from typing import Any
import os
import re

import numpy as np
import torch
from torch import optim
from torch.nn import TripletMarginLoss
from tqdm import tqdm

from app import logger
from app.model import VariableNet
from app.sdfloader import SDFloader


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
    continue_from_checkpoint: int | None = None,
    
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
    total_batches = sdfloader.get_num_batches(batch_size) * num_epochs
    total_batches = sdfloader.get_num_batches(batch_size)
    last_k_losses = []
    moving_avg_losses = []
    last_k_eval_losses = []
    moving_avg_eval_losses = []
    eval_round = False
    x_for_moving_avg = []
    x_for_moving_avg_eval = []
    x = 0
    continue_from_checkpoint = continue_from_checkpoint if continue_from_checkpoint else 0
    for epoch_num in tqdm(range(continue_from_checkpoint, num_epochs), ascii=True, desc="Epoch"):
        start_time = time.time()
        t_loss = 0
        batch_num = 1
        last_time = time.time()
        batch: tuple
        for batch in sdfloader.triplet_batch_generator(batch_size=batch_size):
            x += 1
            if evaluation and batch_num % 10 == 0:
                eval_round = True
                logger.info("Evaluating model")
            else:
                eval_round = False
            if not device == "cpu":
                batch = (
                    batch[0].float().to(device),
                    batch[1].float().to(device),
                    batch[2].float().to(device),
                )
            if eval_round:
                model.eval()
                with torch.no_grad():
                    preds = model(batch)
                model.train()
            else:
                preds = model(batch)
            ancor_preds, positive_preds, negative_preds = preds[0], preds[1], preds[2]
            loss = triplet_loss_module(ancor_preds, positive_preds, negative_preds)
            if not eval_round:
                t_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
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
        if save_checkpoints:
            torch.save(model, checkpoint_dir_path + str(epoch_num) + "_checkpoint.pt")
        
    return model

def load_latest_checkpoint(checkpoint_dir_path: str) -> VariableNet:
    files = os.listdir(checkpoint_dir_path)
    pattern = re.compile(r"\d+_checkpoint\.pt")
    files = [f for f in files if pattern.match(f)]
    if not files:
        return None
    files = sorted(files, key=lambda x: int(x.split("_")[0]))
    latest_checkpoint = files[-1]
    return torch.load(checkpoint_dir_path + latest_checkpoint), int(latest_checkpoint.split("_")[0])

def train(
    sdfloader_path: str,
    output_path: str,
    device: str,
    num_epochs=10,
    batch_size=32,
    save_checkpoints=False,
    checkpoint_dir_path=None,
    evaluation=True,
) -> None:
    if device == "cuda":
        if not torch.cuda.is_available(): 
            raise ValueError("Cuda is not available")
        device = torch.device("cuda")
    
    if save_checkpoints and not checkpoint_dir_path:
        raise ValueError("Checkpoint dir path is required when save_checkpoints is True")

    with open(sdfloader_path, "rb") as f:
        sdfloader: SDFloader = pickle.load(f)
    logger.info("sdfloader_path: " + sdfloader_path)
    logger.info("output_path: " + output_path)
    logger.info("device:" + str(device))
    input_size = sdfloader.sdf.matrix.shape[1]
    logger.info("input size: " + str(input_size))
    continue_from_checkpoint = None
    if not checkpoint_dir_path:
        snn_model = VariableNet(input_size, 2, 320)
        snn_model.to(device)
    else:
        result = load_latest_checkpoint(checkpoint_dir_path)
        if result is None:
            snn_model = VariableNet(input_size, 2, 320)
            snn_model.to(device)
        else:
            snn_model, continue_from_checkpoint = result
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
        continue_from_checkpoint=continue_from_checkpoint,
    )
    torch.save(model, output_path)



