import time
from os import path
from argparse import ArgumentParser

import torch

from torch.utils.data import DataLoader
from torch.nn import NLLLoss, MSELoss
from torch.optim import AdamW

from torchvision.transforms.v2 import (
    Compose, ToDtype, RandomResizedCrop, RandomHorizontalFlip,
    RandomPhotometricDistort,
)

from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.detection import IntersectionOverUnion

from model import VGGDoge, ResDoge34, ResDoge50
from data import DogeDataset

def main():
    parser = ArgumentParser(description='Training script')

    parser.add_argument('--architecture', default='resdoge34', choices=['vggdoge', 'resdoge34', 'resdoge50'], type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--num_epochs', default=1000, type=int)
    parser.add_argument('--eval_epochs', default=10, type=int)
    parser.add_argument('--checkpoint_epochs', default=20, type=int)
    parser.add_argument('--dataset_path', default='./dataset', type=str)
    parser.add_argument('--checkpoint_path', default='./out/ckpt.pt', type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', default='cuda', type=str)

    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError(f'Batch size must be greater than 0, {args.batch_size} given.')

    if args.learning_rate < 0:
        raise ValueError(f'Learning rate must be a positive value, {args.learning_rate} given.')

    if args.num_epochs < 1:
        raise ValueError(f'Must train for at least 1 epoch, {args.num_epochs} given.')

    if args.eval_epochs < 1:
        raise ValueError(f'Eval epochs must be greater than 0, {args.eval_epochs} given.')

    if args.checkpoint_epochs < 1:
        raise ValueError(f'Checkpoint epochs must be greater than 0, {args.checkpoint_epochs} given.')

    if 'cuda' in args.device and not torch.cuda.is_available():
        raise RuntimeError('Cuda is not available.')

    dtype = torch.bfloat16 if args.device == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32

    forward_context = torch.amp.autocast(device_type=args.device, dtype=dtype)

    transformer = Compose([
        RandomResizedCrop((224, 224), ratio=(1, 1.3)),
        RandomHorizontalFlip(),
        RandomPhotometricDistort(),
        ToDtype(torch.float32, scale=True),
    ])

    train = DogeDataset(root_path=args.dataset_path, train=True, transformer=transformer)
    test = DogeDataset(root_path=args.dataset_path, train=False, transformer=transformer)

    num_classes = train.num_classes

    train_loader = DataLoader(train, batch_size=args.batch_size, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test, batch_size=args.batch_size, pin_memory=True)

    match args.architecture:
        case 'resdoge50':
            model = ResDoge50(num_classes)

        case 'resdoge34':
            model = ResDoge34(num_classes)

        case 'vggdoge':
            model = VGGDoge(num_classes)

        case _:
            raise RuntimeError('Invalid network architecture.')

    model = model.to(args.device)

    nll_loss = NLLLoss().to(args.device)
    mse_loss = MSELoss().to(args.device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, fused=True)

    accuracy_metric = MulticlassAccuracy(num_classes, top_k=3).to(args.device)
    iou_metric = IntersectionOverUnion().to(args.device)

    print('Compiling model')
    model = torch.compile(model)

    print(f'Model has {model.num_trainable_params:,} trainable parameters')

    if args.resume:
        checkpoint = torch.load(args.checkpoint_path, map_location=args.device, weights_only=True)

        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        print('Previous checkpoint resumed successfully')

    print('Training ...')
    model.train()

    for epoch in range(1, args.num_epochs + 1):
        total_nll = torch.tensor(0.0).to(args.device)
        total_mse = torch.tensor(0.0).to(args.device)
        total_batches = 0

        start = time.time()

        for x, y1, y2 in train_loader:
            x = x.to(args.device, non_blocking=True)
            y1 = y1.to(args.device, non_blocking=True)
            y2 = y2.to(args.device, non_blocking=True)

            optimizer.zero_grad()

            with forward_context:
                y1_pred, y2_pred = model(x)

                nll = nll_loss(y1_pred, y1)
                mse = mse_loss(y2_pred, y2)

                total_loss = nll / nll.detach() + mse / mse.detach()

            total_loss.backward()

            optimizer.step()

            total_nll += nll
            total_mse += mse

            total_batches += 1

        duration = time.time() - start

        average_cross_entropy = total_nll / total_batches
        average_mse = total_mse / total_batches

        print(
            f'Epoch: {epoch}, Cross Entropy: {average_cross_entropy:.5},',
            f'Box MSE: {average_mse:.7}, Duration: {duration:.2f} seconds'
        )

        if epoch % args.eval_epochs == 0:
            model.eval()

            for x, y1, y2 in test_loader:
                x = x.to(args.device, non_blocking=True)
                y1 = y1.to(args.device, non_blocking=True)
                y2 = y2.to(args.device, non_blocking=True)

                with torch.no_grad():
                    with forward_context:
                        y1_pred, y2_pred = model(x)

                    y2_pred = [{
                        'boxes': box.view(1, -1),
                        'labels': y1[i].view(1),
                    } for i, box in enumerate(y2_pred)]

                    y2_hat = [{
                        'boxes': box.view(1, -1),
                        'labels': y1[i].view(1),
                    } for i, box in enumerate(y2)]

                    accuracy_metric.update(y1_pred, y1)
                    iou_metric.update(y2_pred, y2_hat)

            average_accuracy = accuracy_metric.compute()
            average_iou = iou_metric.compute()

            print(f'Top 3 Accuracy: {average_accuracy:.4}, Box IOU: {average_iou['iou']:.4}')

            accuracy_metric.reset()
            iou_metric.reset()

            model.train()

        if epoch % args.checkpoint_epochs == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'architecture': args.architecture,
            }

            torch.save(checkpoint, args.checkpoint_path)

            print('Checkpoint saved')

if __name__ == '__main__':
    main()