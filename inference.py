import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from core.res_unet import ResUnet
from core.res_unet_plus import ResUnetPlusPlus
from utils.hparams import HParam


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def start_points(size: int, split_size: int, overlap: float = 0.0):
    """根据图像尺寸生成切块起始点。

    逻辑与 preprocess.py 保持一致：
    - stride = split_size * (1 - overlap)
    - 最后一个切块强制贴边，保证覆盖到图像右下角
    """
    if size <= split_size:
        # 小图无需切块，直接从 0 开始
        return [0]

    points = [0]
    stride = int(split_size * (1 - overlap))
    stride = max(stride, 1)
    counter = 1

    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        points.append(pt)
        counter += 1

    return points


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="ResUnet inference")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="yaml file for configuration"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="path to model checkpoint"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="input image path or directory containing images",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="output file path (single image mode) or output directory (directory mode)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="binary mask threshold, default: 0.5",
    )
    parser.add_argument(
        "--save-prob",
        action="store_true",
        help="save probability map alongside binary mask",
    )
    parser.add_argument(
        "--resnet-plus-plus",
        action="store_true",
        help="force using ResUnet++ model (otherwise follows config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help='device override, e.g. "cpu" or "cuda:0"',
    )
    parser.add_argument(
        "--tile-overlap",
        type=float,
        default=0.14,
        help="tile overlap ratio for large images, default: 0.14",
    )
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    """返回推理设备。

    优先使用命令行指定设备，否则自动选择 CUDA/CPU。
    """
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(hp, checkpoint_path: str, device: torch.device, force_resunetpp: bool):
    """加载模型结构与权重。

    支持两种 checkpoint 形态：
    - 训练脚本保存的字典（包含 state_dict）
    - 直接保存的纯 state_dict
    """
    # 优先使用命令行强制参数，否则读取配置文件开关
    use_resunetpp = force_resunetpp or bool(hp.RESNET_PLUS_PLUS)
    if use_resunetpp:
        model = ResUnetPlusPlus(3)
    else:
        model = ResUnet(3)

    # map_location 确保不同设备间加载时不会报错
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model, use_resunetpp


def collect_input_images(input_path: Path):
    """收集输入图像列表。

    - 若输入是单文件：返回长度为 1 的列表
    - 若输入是目录：递归收集支持的图像后缀文件
    """
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    if not input_path.is_dir():
        raise ValueError(f"Unsupported input path: {input_path}")

    images = [
        p
        for p in sorted(input_path.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not images:
        raise ValueError(f"No image files found in: {input_path}")
    return images


def preprocess_image(image_path: Path, device: torch.device):
    """读取并转换单张图片为模型输入张量。

    输出张量维度为 [1, 3, H, W]，值域为 [0, 1]。
    """
    image = Image.open(image_path).convert("RGB")
    tensor = TF.to_tensor(image).unsqueeze(0).to(device)
    return image, tensor


def predict_with_tiling(model, img_tensor, crop_size: int, overlap: float):
    """对单张图进行推理，自动处理超尺寸图片。

    - 小图：直接整图推理
    - 大图：按切块滑窗推理，并在重叠区做均值融合
    """
    _, _, h, w = img_tensor.shape

    if h <= crop_size and w <= crop_size:
        # 小图直接前向
        output = model(img_tensor)
        if output.min().item() < 0 or output.max().item() > 1:
            # 兼容输出非概率值的模型
            output = torch.sigmoid(output)
        return output

    # 大图切块：x/y 方向起点与 preprocess 的裁剪策略一致
    x_points = start_points(w, crop_size, overlap)
    y_points = start_points(h, crop_size, overlap)

    # output_sum: 累积每个像素的预测值
    # output_count: 记录每个像素被覆盖次数，用于重叠区域平均
    output_sum = torch.zeros((1, 1, h, w), device=img_tensor.device)
    output_count = torch.zeros((1, 1, h, w), device=img_tensor.device)

    for y in y_points:
        for x in x_points:
            print(f"Processing tile: x={x}, y={y}, size={crop_size}")
            tile = img_tensor[:, :, y : y + crop_size, x : x + crop_size]
            tile_output = model(tile)
            if tile_output.min().item() < 0 or tile_output.max().item() > 1:
                tile_output = torch.sigmoid(tile_output)

            tile_h = tile_output.shape[2]
            tile_w = tile_output.shape[3]
                # 将 tile 结果累积回原图对应位置
            output_sum[:, :, y : y + tile_h, x : x + tile_w] += tile_output
            output_count[:, :, y : y + tile_h, x : x + tile_w] += 1

            # 重叠区域取均值，得到平滑且完整的整图预测
    return output_sum / output_count.clamp(min=1)


def save_outputs(prob_map: torch.Tensor, out_prefix: Path, threshold: float, save_prob: bool):
    """保存概率图与二值分割图。

    文件命名规则：
    - 二值图：*_mask.png
    - 概率图：*_prob.png（当 --save-prob 开启时）
    """
    # [1, 1, H, W] -> [H, W]
    prob_np = prob_map.squeeze(0).squeeze(0).cpu().numpy()
    prob_np = np.clip(prob_np, 0.0, 1.0)

    # 按阈值生成二值分割图
    mask_np = (prob_np >= threshold).astype(np.uint8) * 255
    mask_image = Image.fromarray(mask_np)
    mask_path = out_prefix.with_name(out_prefix.name + "_mask.png")
    mask_image.save(mask_path)

    prob_path = None
    if save_prob:
        prob_img = (prob_np * 255).astype(np.uint8)
        prob_image = Image.fromarray(prob_img)
        prob_path = out_prefix.with_name(out_prefix.name + "_prob.png")
        prob_image.save(prob_path)

    return mask_path, prob_path


def run_inference(
    model,
    image_paths,
    input_path: Path,
    output_path: Path,
    device,
    threshold,
    save_prob,
    crop_size: int,
    tile_overlap: float,
):
    """执行推理主流程，支持单图和目录批量模式。"""
    single_file_mode = input_path.is_file()
    if single_file_mode:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for image_path in image_paths:
            _, img_tensor = preprocess_image(image_path, device)
            output = predict_with_tiling(model, img_tensor, crop_size, tile_overlap)

            if single_file_mode:
                # 单图模式：输出前缀来自 --output 文件名
                out_prefix = output_path.with_suffix("")
            else:
                # 目录模式：保持输入目录的相对层级
                relative_path = image_path.relative_to(input_path)
                output_stem = output_path / relative_path.parent / relative_path.stem
                output_stem.parent.mkdir(parents=True, exist_ok=True)
                out_prefix = output_stem

            mask_path, prob_path = save_outputs(output, out_prefix, threshold, save_prob)

            if prob_path:
                print(f"{image_path} -> mask: {mask_path}, prob: {prob_path}")
            else:
                print(f"{image_path} -> mask: {mask_path}")


def main():
    """脚本入口：参数解析、模型加载、推理执行。"""
    args = parse_args()

    hp = HParam(args.config)
    input_path = Path(args.input)
    output_path = Path(args.output)
    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = get_device(args.device)
    image_paths = collect_input_images(input_path)

    model, use_resunetpp = load_model(
        hp,
        str(checkpoint_path),
        device,
        args.resnet_plus_plus,
    )

    print(f"Using device: {device}")
    print(
        "Loaded model: {}".format(
            "ResUnetPlusPlus" if use_resunetpp else "ResUnet"
        )
    )
    print(f"Tile size: {hp.CROP_SIZE}, overlap: {args.tile_overlap}")
    print(f"Found {len(image_paths)} image(s) for inference")

    run_inference(
        model,
        image_paths,
        input_path,
        output_path,
        device,
        args.threshold,
        args.save_prob,
        int(hp.CROP_SIZE),
        args.tile_overlap,
    )


if __name__ == "__main__":
    main()