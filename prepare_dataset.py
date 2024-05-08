# Author: Matej Sirovatka
# Based on: https://github.com/princeton-vl/RAFT/blob/master/demo.py

from pathlib import Path
import sys  # isort:skip

sys.path.append("core")

from utils.utils import InputPadder
from utils import flow_viz
from raft import RAFT
from PIL import Image
import torch
import numpy as np
import os
import argparse
from torchvision.utils import save_image
from tqdm import tqdm

DEVICE = "cuda"


def load_image(imfile: str):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def convert_to_img(flow: torch.Tensor) -> torch.Tensor:
    flow = flow[0].permute(1, 2, 0).cpu().numpy()

    flow = flow_viz.flow_to_image(flow)

    flow_tensor = torch.from_numpy(flow).permute(2, 0, 1) / 255.0
    return flow_tensor


@torch.no_grad()
def convert_dataset(args):
    model = torch.nn.DataParallel(RAFT(args), device_ids=args.devices)
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    dataset_path = Path(args.dataset_path)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    for seq_path in tqdm(list(dataset_path.iterdir())):
        output_dir = Path(args.outdir) / seq_path.name
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        first_frame_iter = sorted(list(seq_path.iterdir()))[:]
        second_frame_iter = sorted(list(seq_path.iterdir()))[1:]
        third_frame_iter = sorted(list(seq_path.iterdir()))[2:]
        for first_frame, second_frame, third_frame in zip(first_frame_iter, second_frame_iter, third_frame_iter):
            image1 = load_image(first_frame)
            image2 = load_image(second_frame)
            image3 = load_image(third_frame)
            padder = InputPadder(image1.shape)
            image1, image2, image3 = padder.pad(image1, image2, image3)
            print(image1)
            print(image2.shape)

            _, flow_up = model(image2, image3, iters=20, test_mode=True)
            _, flow_down = model(image2, image1, iters=20, test_mode=True)

            up_output_path = Path(args.outdir) / seq_path.name / (second_frame.name.removesuffix(".png") + "_up")
            down_output_path = Path(args.outdir) / seq_path.name / (second_frame.name.removesuffix(".png") + "_down")

            torch.save(flow_up, up_output_path.with_suffix(".pt"))
            torch.save(flow_down, down_output_path.with_suffix(".pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="restore checkpoint")
    parser.add_argument("--small", action="store_true", help="use small model")
    parser.add_argument("--mixed_precision", action="store_true", help="use mixed precision")
    parser.add_argument("--alternate_corr", action="store_true", help="use efficent correlation implementation")
    parser.add_argument("--devices", type=int, nargs="+", default=[0], help="devices to use")

    parser.add_argument("--outdir", type=str, default="demo_flows", help="output directory for saving flows")
    parser.add_argument("--dataset-path", type=str, default="data/frames", help="path to the dataset")
    args = parser.parse_args()

    convert_dataset(args)
