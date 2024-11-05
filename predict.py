from os import path
from argparse import ArgumentParser

import torch

from torchvision.io import decode_image
from torchvision.transforms.v2 import Compose, Resize, CenterCrop, ToDtype

from model import ResDoge50, ResDoge34, VGGDoge

from data import DogeDataset

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


def main():
    parser = ArgumentParser(description="Inference script")

    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--top_k", default=3, type=int)
    parser.add_argument("--checkpoint_path", default="./out/ckpt.pt", type=str)
    parser.add_argument("--device", default="cuda", type=str)

    args = parser.parse_args()

    if args.top_k < 1 or args.top_k > 120:
        raise ValueError(f"K must be between 1 and 120, {args.top_k} given.")

    if "cuda" in args.device and not torch.cuda.is_available():
        raise RuntimeError("Cuda is not available.")

    dtype = (
        torch.bfloat16
        if args.device == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )

    forward_context = torch.amp.autocast(device_type=args.device, dtype=dtype)

    checkpoint = torch.load(
        args.checkpoint_path, map_location=args.device, weights_only=True
    )

    num_classes = len(DogeDataset.CLASS_MAPPING)

    match checkpoint["architecture"]:
        case "resdoge50":
            model = ResDoge50(num_classes)

        case "resdoge34":
            model = ResDoge34(num_classes)

        case "vggdoge":
            model = VGGDoge(num_classes)

        case _:
            raise RuntimeError("Invalid network architecture.")

    model = model.to(args.device)

    print("Compiling model")
    model = torch.compile(model)

    model.load_state_dict(checkpoint["model"])

    print("Model checkpoint loaded successfully")

    transformer = Compose(
        [
            Resize(224),
            CenterCrop((224, 224)),
            ToDtype(torch.float32, scale=True),
        ]
    )

    image = decode_image(args.image_path, mode="RGB")

    image = transformer(image)

    x = image.unsqueeze(0).to(args.device)

    model.eval()

    print("Predicting ...")

    with torch.no_grad():
        with forward_context:
            y1_pred, y2_pred = model(x)

        y1_pred = torch.exp(y1_pred)

    probabilities, indices = torch.topk(y1_pred[0], k=args.top_k)

    predictions = [DogeDataset.CLASS_MAPPING[index] for index in indices]

    label = predictions[0]

    box = y2_pred[0].tolist()

    print(f"Top {args.top_k}:")
    for i in range(0, args.top_k):
        print(f"{predictions[i]}, Probability: {probabilities[i]:.4f}")

    image = image.permute(1, 2, 0)  # C x W x H -> W x H x C

    xmin, ymin, xmax, ymax = box

    xy, width, height = (xmin, ymin), xmax - xmin, ymax - ymin

    patch = Rectangle(xy, width, height, linewidth=2, edgecolor="r", facecolor="none")

    figure, axis = plt.subplots()

    axis.imshow(image)
    axis.add_patch(patch)
    axis.text(xmin + 3, ymax - 3, label, fontsize=18, color="r")

    plt.show()


if __name__ == "__main__":
    main()
