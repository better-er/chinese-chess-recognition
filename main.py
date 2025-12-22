import sys
from pathlib import Path
from typing import Tuple

import requests
import click
from loguru import logger
import pyperclip

def _download_onnx_if_needed(dst: Path) -> Path:
    if dst.exists():
        return dst
    
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    url = (
        "https://huggingface.co/spaces/yolo12138/Chinese_Chess_Recognition/resolve/main/"
        "onnx/layout_recognition/nano_v3-0319.onnx?download=true"
    )
    
    logger.info(f"Downloading ONNX model to {dst} ...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()  # 确保请求成功
    
    with open(dst, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    
    return dst


def _classes() -> Tuple[str, ...]:
    # Keep the same order as config dict_cate_names
    return (
        '.', 'x', 'K', 'A', 'B', 'N', 'R', 'C', 'P',
        'k', 'a', 'b', 'n', 'r', 'c', 'p',
    )


def run_onnx_infer(img_path: Path, onnx_path: Path) -> str:
    import cv2
    import numpy as np
    import onnxruntime as ort

    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"无法加载图像: {img_path}")

    # preprocess: BGR->RGB, resize 256, normalize
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
    rgb = rgb.astype(np.float32)
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    rgb = (rgb - mean) / std
    chw = np.transpose(rgb, (2, 0, 1))  # HWC -> CHW
    inp = np.expand_dims(chw, axis=0)  # 1x3x256x256

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out = sess.run(None, {in_name: inp})
    logits = out[0]  # assume shape: 1x90x16 or 1x16x10x9 or similar

    # normalize to (90, 16)
    arr = np.array(logits)
    if arr.ndim == 4:
        if arr.shape[1:] == (16, 10, 9):
            arr = np.transpose(arr, (0, 2, 3, 1))  # -> 1x10x9x16
        elif arr.shape[1:] != (10, 9, 16):
            raise RuntimeError(f"Unexpected output shape: {arr.shape}")
        arr = arr.reshape(-1, 16)
    elif arr.ndim == 3:
        if arr.shape[1:] == (90, 16):
            arr = arr.reshape(-1, 16)
        elif arr.shape[1:] == (16, 90):
            arr = np.transpose(arr, (0, 2, 1)).reshape(-1, 16)
        else:
            raise RuntimeError(f"Unexpected output shape: {arr.shape}")
    else:
        raise RuntimeError(f"Unexpected output ndim: {arr.ndim}")

    cls_idx = arr.argmax(axis=1)
    classes = _classes()
    labels = [classes[i] for i in cls_idx.tolist()]

    # build FEN-like rows (10x9)
    rows, cols = 10, 9
    fen_rows = []
    for r in range(rows):
        row = labels[r * cols:(r + 1) * cols]
        empty = 0
        parts = []
        for ch in row:
            if ch in ('.', 'x'):
                empty += 1
            else:
                if empty:
                    parts.append(str(empty))
                    empty = 0
                parts.append(ch)
        if empty:
            parts.append(str(empty))
        fen_rows.append(''.join(parts))
    fen = '/'.join(fen_rows)
    return fen


def _recognize(img: Path | None, onnx: Path, first: str) -> str:
    root = Path(__file__).resolve().parent
    logger.info("✅ Chinese Chess Recognition: main.py running")
    logger.info(f"Python: {sys.version.split()[0]}")
    logger.info(f"Workspace: {root}")

    if img is None:
        img_path = root / "assets" / "demo.jpg"
        from PIL import Image, ImageGrab

        clip = ImageGrab.grabclipboard()
        if clip is None:
            logger.error("❌ 剪切板中未找到图片或文件路径")
            print("{'error': '剪切板中未找到图片或文件路径'}")
            sys.exit(1)

        if isinstance(clip, Image.Image):
            clip.save(img_path)
            logger.info(f"💾 从剪切板保存图像到: {img_path}")
        elif isinstance(clip, (list, tuple)) and len(clip) > 0:
            first_path = Path(clip[0])
            if not first_path.exists():
                logger.error(f"❌ 剪切板中的文件路径不存在: {first_path}")
                print(f"{{'error': '剪切板中的文件路径不存在: {first_path}'}}")
                sys.exit(1)
            img = Image.open(first_path)
            img.save(img_path)
            logger.info(f"💾 从文件路径 {first_path} 保存图像到: {img_path}")
        else:
            logger.error("❌ 剪切板内容不是图像或可打开的文件路径")
            print("{'error': '剪切板内容不是图像或可打开的文件路径'}")
            sys.exit(1)
    else:
        img_path = img
        if not img_path.exists():
            logger.error(f"❌ 图像文件不存在: {img_path}")
            print(f"{{'error': '图像文件不存在: {img_path}'}}")
            sys.exit(1)

    onnx_path = _download_onnx_if_needed(onnx)
    fen = run_onnx_infer(img_path, onnx_path)

    logger.info("FEN表示：")
    logger.info(f"{fen} w - - 0 1")
    logger.info(f"{fen} b - - 0 1")

    # 返回完整 FEN，供外部调用者复用
    return f"{fen} {first} - - 0 1"


@click.command(help="识别中国象棋棋盘布局，输出 FEN 表示并复制到剪切板。[w|b]表示先手方（默认 w）")
@click.option(
    "--img",
    type=click.Path(path_type=Path),
    default=None,
    help="输入图像路径；若缺省则从剪切板获取",
)
@click.option(
    "--onnx",
    type=click.Path(path_type=Path),
    default=Path("cchess_reg/checkpoints/nano_v3-0319.onnx"),
    help="ONNX 模型路径",
)
@click.argument(
    "first",
    type=click.Choice(["w", "b"]),
    default="w",  # 默认值，但用户未提供时才用
    required=False,  # 允许不传，使用默认值
    nargs=1,  # 仅一个参数
)
def main(img: Path | None, onnx: Path, first: str) -> str:
    fen_with_side = _recognize(img, onnx, first)

    # 检查 pyperclip 是否可用（不使用 try-except）
    pyperclip.copy(fen_with_side)
    logger.info("✅ 已复制到剪切板")

    return fen_with_side


if __name__ == "__main__":
    main()
