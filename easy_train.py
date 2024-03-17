import argparse

from app import get_cfg
from app.train import train

if __name__ == "__main__":
    cfg = get_cfg()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sdfloader_path", type=str, required=True, help="[REQUIRED] Path to sdfloader"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="[REQUIRED] output pytorch checkpoints and model dir path",
    )
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--device",
        type=str,
        help="Device - cpu or torch device - if not used, will use user_device in cfg.yaml",
    )
    args = parser.parse_args()
    if not args.device:
        args.device = get_cfg()["user_device"]
    num_epochs = args.num_epochs if args.num_epochs else 10
    batch_size = args.batch_size if args.batch_size else 32
    output_dir = args.output_dir
    if output_dir[-1] != "/":
        output_dir += "/"
    num_workers = cfg["nthreads"]
    train(
        sdfloader_path=args.sdfloader_path,
        output_path=args.output_dir + "model.pt",
        device=args.device,
        num_epochs=num_epochs,
        batch_size=batch_size,
        save_checkpoints=True,
        checkpoint_dir_path=args.output_dir,
        evaluation=True,
        num_workers=num_workers
    )
