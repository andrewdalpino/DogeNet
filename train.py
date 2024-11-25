import random

from os import path
from argparse import ArgumentParser

import torch

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import autocast
from torch.cuda import is_available as cuda_is_available, is_bf16_supported

from torchvision.transforms.v2 import (
    Compose,
    ToDtype,
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomPhotometricDistort,
    Resize,
    CenterCrop,
)

from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.detection import IntersectionOverUnion

from model import VGGDoge, ResDoge34, ResDoge50
from data import DogeDataset

from tqdm import tqdm


def main():
    parser = ArgumentParser(description="Training script")

    parser.add_argument(
        "--architecture",
        default="resdoge34",
        choices=["vggdoge", "resdoge34", "resdoge50"],
        type=str,
    )
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--learning_rate", default=5e-4, type=float)
    parser.add_argument("--num_epochs", default=1000, type=int)
    parser.add_argument("--eval_interval", default=10, type=int)
    parser.add_argument("--dataset_path", default="./dataset", type=str)
    parser.add_argument("--checkpoint_interval", default=20, type=int)
    parser.add_argument("--checkpoint_path", default="./out/checkpoint.pt", type=str)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=None, type=int)

    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError(f"Batch size must be greater than 0, {args.batch_size} given.")

    if args.learning_rate < 0:
        raise ValueError(
            f"Learning rate must be a positive value, {args.learning_rate} given."
        )

    if args.num_epochs < 1:
        raise ValueError(f"Must train for at least 1 epoch, {args.num_epochs} given.")

    if args.eval_interval < 1:
        raise ValueError(
            f"Eval interval must be greater than 0, {args.eval_interval} given."
        )

    if args.checkpoint_interval < 1:
        raise ValueError(
            f"Checkpoint interval must be greater than 0, {args.checkpoint_interval} given."
        )

    if "cuda" in args.device and not torch.cuda.is_available():
        raise RuntimeError("Cuda is not available.")

    dtype = (
        torch.bfloat16
        if args.device == "cuda" and is_bf16_supported()
        else torch.float32
    )

    forward_context = autocast(device_type=args.device, dtype=dtype)

    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_transformer = Compose(
        [
            RandomResizedCrop((224, 224), ratio=(1, 1.3)),
            RandomHorizontalFlip(),
            RandomPhotometricDistort(),
            ToDtype(torch.float32, scale=True),
        ]
    )
    test_transformer = Compose(
        [
            Resize(224),
            CenterCrop((224, 224)),
            ToDtype(torch.float32, scale=True),
        ]
    )

    training = DogeDataset(
        root_path=args.dataset_path, train=True, transformer=train_transformer
    )
    testing = DogeDataset(
        root_path=args.dataset_path, train=False, transformer=test_transformer
    )

    train_loader = DataLoader(
        training,
        batch_size=args.batch_size,
        pin_memory="cpu" not in args.device,
        shuffle=True,
    )
    test_loader = DataLoader(
        testing,
        batch_size=args.batch_size,
        pin_memory="cpu" not in args.device,
        shuffle=False,
    )

    model_args = {
        "num_classes": training.num_classes,
    }

    match args.architecture:
        case "resdoge50":
            model = ResDoge50(**model_args)

        case "resdoge34":
            model = ResDoge34(**model_args)

        case "vggdoge":
            model = VGGDoge(**model_args)

        case _:
            raise RuntimeError("Invalid network architecture.")

    model = model.to(args.device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, fused=True)

    accuracy_metric = MulticlassAccuracy(training.num_classes, top_k=3).to(args.device)
    iou_metric = IntersectionOverUnion().to(args.device)

    print("Compiling model")
    model = torch.compile(model)

    print(f"Model has {model.num_trainable_params:,} trainable parameters")

    if args.resume:
        checkpoint = torch.load(
            args.checkpoint_path, map_location=args.device, weights_only=True
        )

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        print("Previous checkpoint resumed successfully")

    print("Training ...")
    model.train()

    for epoch in range(1, args.num_epochs + 1):
        total_nll, total_mse = 0.0, 0.0
        total_batches = 0

        for x, y1, y2 in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            x = x.to(args.device, non_blocking=True)
            y1 = y1.to(args.device, non_blocking=True)
            y2 = y2.to(args.device, non_blocking=True)

            with forward_context:
                y1_pred, y2_pred, nll, mse = model(x, y1, y2)

                total_loss = nll / nll.detach() + mse / mse.detach()

            total_loss.backward()

            optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            total_nll += nll.item()
            total_mse += mse.item()

            total_batches += 1

        average_cross_entropy = total_nll / total_batches
        average_mse = total_mse / total_batches

        print(
            f"Epoch {epoch}:",
            f"Cross Entropy: {average_cross_entropy:.5},",
            f"Box MSE: {average_mse:.7}",
        )

        if epoch % args.eval_interval == 0:
            model.eval()

            for x, y1, y2 in tqdm(test_loader, desc="Testing", leave=False):
                x = x.to(args.device, non_blocking=True)
                y1 = y1.to(args.device, non_blocking=True)
                y2 = y2.to(args.device, non_blocking=True)

                with torch.no_grad():
                    with forward_context:
                        y1_pred, y2_pred, _, _ = model(x)

                    y2_pred = [
                        {
                            "boxes": box.unsqueeze(0),
                            "labels": label.unsqueeze(0),
                        }
                        for box, label in zip(y2_pred, y1)
                    ]
                    y2 = [
                        {
                            "boxes": box.unsqueeze(0),
                            "labels": label.unsqueeze(0),
                        }
                        for box, label in zip(y2, y1)
                    ]

                    accuracy_metric.update(y1_pred, y1)
                    iou_metric.update(y2_pred, y2)

            average_accuracy = accuracy_metric.compute()
            average_iou = iou_metric.compute()

            print(
                f"Top 3 Accuracy: {average_accuracy:.3},",
                f"Box IOU: {average_iou['iou']:.4}",
            )

            accuracy_metric.reset()
            iou_metric.reset()

            model.train()

        if epoch % args.checkpoint_interval == 0:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "architecture": args.architecture,
                "model_args": model_args,
            }

            torch.save(checkpoint, args.checkpoint_path)

            print("Checkpoint saved")


if __name__ == "__main__":
    main()
