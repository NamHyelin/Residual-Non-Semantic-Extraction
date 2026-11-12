import argparse
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import cv2
from PIL import Image

# ==============================
# Optional dependencies
# ==============================
# pip install pillow numpy torch torchvision transformers diffusers controlnet-aux
# pip install open_clip_torch pytorch-msssim accelerate safetensors

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
except Exception:
    BlipProcessor = None
    BlipForConditionalGeneration = None

try:
    from controlnet_aux import HEDdetector
except Exception:
    HEDdetector = None

try:
    from diffusers import (
        StableDiffusionControlNetPipeline,
        ControlNetModel,
        UniPCMultistepScheduler,
    )
except Exception:
    StableDiffusionControlNetPipeline = None
    ControlNetModel = None
    UniPCMultistepScheduler = None

try:
    import open_clip
except Exception:
    open_clip = None

try:
    from pytorch_msssim import ms_ssim
except Exception:
    ms_ssim = None


# ==============================
# Config / Data classes
# ==============================
@dataclass
class ChannelConfig:
    snr_db_text: float = 8.0
    snr_db_edge: float = 5.0
    seed: int = 42


@dataclass
class GenerationConfig:
    image_size: int = 512
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    negative_prompt: str = "low quality, blurry, distorted"
    use_half: bool = True


@dataclass
class RunResult:
    caption: str
    text_packet_original: str
    text_packet_received: str
    text_bit_error_rate: float
    edge_bit_error_rate: float
    clip_score: Optional[float]
    ms_ssim_score: Optional[float]
    output_image_path: str
    edge_map_path: str
    report_path: str


# ==============================
# Utility
# ==============================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_image_rgb(path: str, size: Optional[int] = None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.Resampling.LANCZOS)
    return img


def save_image(img: Image.Image, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


def pil_to_uint8_np(img: Image.Image) -> np.ndarray:
    return np.array(img).astype(np.uint8)


def bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = np.asarray(bits, dtype=np.uint8)
    pad = (-len(bits)) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits).tobytes()


def text_to_bits(text: str) -> np.ndarray:
    return bytes_to_bits(text.encode("utf-8", errors="replace"))


def bits_to_text(bits: np.ndarray) -> str:
    raw = bits_to_bytes(bits)
    return raw.decode("utf-8", errors="replace")


def image_to_bits_gray(img: Image.Image) -> np.ndarray:
    gray = img.convert("L")
    arr = np.array(gray, dtype=np.uint8)
    return np.unpackbits(arr.flatten())


def bits_to_image_gray(bits: np.ndarray, size: Tuple[int, int]) -> Image.Image:
    w, h = size
    n_pixels = w * h
    n_bits = n_pixels * 8
    bits = np.asarray(bits[:n_bits], dtype=np.uint8)
    if len(bits) < n_bits:
        bits = np.concatenate([bits, np.zeros(n_bits - len(bits), dtype=np.uint8)])
    arr = np.packbits(bits).reshape((h, w)).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def bpsk_awgn_transmit(
    bits: np.ndarray, snr_db: float, rng: np.random.Generator
) -> Tuple[np.ndarray, float]:
    """
    Simple uncoded BPSK over AWGN.
    0 -> -1, 1 -> +1
    sigma^2 = 1 / (2 * snr_linear)
    """
    bits = np.asarray(bits, dtype=np.uint8)
    symbols = 2.0 * bits.astype(np.float32) - 1.0
    snr_linear = 10 ** (snr_db / 10.0)
    sigma = math.sqrt(1.0 / (2.0 * snr_linear))
    noise = rng.normal(0.0, sigma, size=symbols.shape).astype(np.float32)
    received = symbols + noise
    decoded = (received >= 0).astype(np.uint8)
    ber = float(np.mean(decoded != bits)) if len(bits) > 0 else 0.0
    return decoded, ber


def safe_filename(text: str, max_len: int = 80) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in text)
    return cleaned[:max_len].strip("_") or "output"


# ==============================
# Semantic extractor: caption
# ==============================
class CaptionExtractor:
    def __init__(self, device: str):
        if BlipProcessor is None or BlipForConditionalGeneration is None:
            raise ImportError(
                "transformers BLIP is not installed. Please install transformers."
            )
        self.device = device
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def extract(self, image: Image.Image) -> str:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=40)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption.strip()


# ==============================
# Semantic extractor: HED edge
# ==============================
class EdgeExtractor:
    def __init__(self):
        pass

    def extract(self, image: Image.Image, out_size: int = 512) -> Image.Image:
        image = image.resize((out_size, out_size), Image.Resampling.LANCZOS)
        arr = np.array(image)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        edge = cv2.Canny(gray, 100, 200)
        return Image.fromarray(edge).convert("L")


# ==============================
# Generative decoder: ControlNet
# ==============================
class SemanticDecoder:
    def __init__(self, device: str, cfg: GenerationConfig):
        try:
            from diffusers import StableDiffusionControlNetPipeline
            from diffusers import ControlNetModel
            from diffusers import UniPCMultistepScheduler
        except Exception:
            raise ImportError("diffusers is not installed. Please install diffusers.")

        dtype = (
            torch.float16
            if (cfg.use_half and device.startswith("cuda"))
            else torch.float32
        )
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny", torch_dtype=dtype
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe = self.pipe.to(device)
        if device.startswith("cuda"):
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        self.device = device
        self.cfg = cfg

    @torch.no_grad()
    def generate(self, prompt: str, edge_map: Image.Image) -> Image.Image:
        image = self.pipe(
            prompt=prompt,
            negative_prompt=self.cfg.negative_prompt,
            image=edge_map.convert("RGB"),
            num_inference_steps=self.cfg.num_inference_steps,
            guidance_scale=self.cfg.guidance_scale,
        ).images[0]
        return image


# ==============================
# Metrics
# ==============================
def compute_clip_score(
    image_a: Image.Image, image_b: Image.Image, device: str
) -> Optional[float]:
    if open_clip is None:
        return None
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        a = preprocess(image_a).unsqueeze(0).to(device)
        b = preprocess(image_b).unsqueeze(0).to(device)
        fa = model.encode_image(a)
        fb = model.encode_image(b)
        fa = fa / fa.norm(dim=-1, keepdim=True)
        fb = fb / fb.norm(dim=-1, keepdim=True)
        score = (fa * fb).sum(dim=-1).item()
    return float(score)


def compute_ms_ssim_score(
    image_a: Image.Image, image_b: Image.Image, device: str
) -> Optional[float]:
    if ms_ssim is None:
        return None
    a = (
        torch.from_numpy(np.array(image_a).astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )
    b = (
        torch.from_numpy(np.array(image_b).astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )
    a = a.to(device)
    b = b.to(device)
    with torch.no_grad():
        score = ms_ssim(a, b, data_range=1.0, size_average=True).item()
    return float(score)


# ==============================
# Main pipeline
# ==============================
def build_received_edge_map(
    edge_img: Image.Image,
    snr_db: float,
    seed: int,
) -> Tuple[Image.Image, float]:
    bits = image_to_bits_gray(edge_img)
    rng = np.random.default_rng(seed)
    rx_bits, ber = bpsk_awgn_transmit(bits, snr_db, rng)
    rx_img = bits_to_image_gray(rx_bits, edge_img.size)
    return rx_img, ber


def build_received_text(
    caption: str,
    snr_db: float,
    seed: int,
) -> Tuple[str, float]:
    bits = text_to_bits(caption)
    rng = np.random.default_rng(seed)
    rx_bits, ber = bpsk_awgn_transmit(bits, snr_db, rng)
    rx_text = bits_to_text(rx_bits)
    return rx_text, ber


def maybe_recover_prompt(original_caption: str, received_caption: str) -> str:
    """
    Very simple robustness rule.
    If the received prompt becomes unreadable, fall back to the original.
    For a pure communication experiment, you may want to remove this fallback.
    """
    txt = received_caption.strip()
    if len(txt) < 3:
        return original_caption
    bad_ratio = sum(1 for c in txt if c == "\ufffd") / max(len(txt), 1)
    if bad_ratio > 0.2:
        return original_caption
    return txt


def run_pipeline(
    input_image_path: str,
    output_dir: str,
    channel_cfg: ChannelConfig,
    gen_cfg: GenerationConfig,
) -> RunResult:
    os.makedirs(output_dir, exist_ok=True)
    set_seed(channel_cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # 1) Load image
    image = load_image_rgb(input_image_path, size=gen_cfg.image_size)

    # 2) Semantic encoding
    print("[INFO] Loading caption extractor...")
    caption_extractor = CaptionExtractor(device)
    print("[INFO] Loading edge extractor...")
    edge_extractor = EdgeExtractor()

    print("[INFO] Extracting caption...")
    caption = caption_extractor.extract(image)
    print(f"[INFO] Caption: {caption}")

    print("[INFO] Extracting HED edge map...")
    edge_map = edge_extractor.extract(image, out_size=gen_cfg.image_size)
    edge_map_path = os.path.join(output_dir, "edge_map.png")
    save_image(edge_map, edge_map_path)

    # 3) Transmission (uncoded BPSK simulation)
    print("[INFO] Simulating text-channel...")
    rx_caption, text_ber = build_received_text(
        caption, channel_cfg.snr_db_text, channel_cfg.seed + 1
    )
    print(f"[INFO] Received caption: {rx_caption}")
    print(f"[INFO] Text BER: {text_ber:.6f}")

    print("[INFO] Simulating edge-channel...")
    rx_edge_map, edge_ber = build_received_edge_map(
        edge_map, channel_cfg.snr_db_edge, channel_cfg.seed + 2
    )
    rx_edge_path = os.path.join(output_dir, "received_edge_map.png")
    save_image(rx_edge_map, rx_edge_path)
    print(f"[INFO] Edge BER: {edge_ber:.6f}")

    # 4) Generative decoding
    robust_prompt = maybe_recover_prompt(caption, rx_caption)
    print("[INFO] Loading semantic decoder (ControlNet + SD)...")
    decoder = SemanticDecoder(device, gen_cfg)

    print("[INFO] Generating reconstructed image...")
    out_img = decoder.generate(prompt=robust_prompt, edge_map=rx_edge_map)
    output_image_path = os.path.join(output_dir, "reconstructed.png")
    save_image(out_img, output_image_path)

    # 5) Evaluation
    print("[INFO] Computing CLIP / MS-SSIM...")
    clip_score = compute_clip_score(image, out_img, device)
    ms_ssim_score = compute_ms_ssim_score(image, out_img, device)

    report = {
        "input_image": input_image_path,
        "device": device,
        "caption_original": caption,
        "caption_received": rx_caption,
        "prompt_used_for_decoding": robust_prompt,
        "text_ber": text_ber,
        "edge_ber": edge_ber,
        "clip_score": clip_score,
        "ms_ssim_score": ms_ssim_score,
        "snr_db_text": channel_cfg.snr_db_text,
        "snr_db_edge": channel_cfg.snr_db_edge,
        "num_inference_steps": gen_cfg.num_inference_steps,
        "guidance_scale": gen_cfg.guidance_scale,
        "image_size": gen_cfg.image_size,
        "artifacts": {
            "edge_map": edge_map_path,
            "received_edge_map": rx_edge_path,
            "reconstructed": output_image_path,
        },
    }

    report_path = os.path.join(output_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return RunResult(
        caption=caption,
        text_packet_original=caption,
        text_packet_received=rx_caption,
        text_bit_error_rate=text_ber,
        edge_bit_error_rate=edge_ber,
        clip_score=clip_score,
        ms_ssim_score=ms_ssim_score,
        output_image_path=output_image_path,
        edge_map_path=edge_map_path,
        report_path=report_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prototype implementation of a semantic communication image pipeline."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="C:/Users/hyeli/Downloads/drive-download-20250625T050437Z-1-001/img_ref1.jpg",
        help="Path to input image",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="C:/TransUnrecon/VAE_MNIST_2/GSemComm",
        help="Output directory",
    )
    parser.add_argument(
        "--snr_text", type=float, default=50.0, help="SNR(dB) for text stream" 
    )
    parser.add_argument(
        "--snr_edge", type=float, default=50.0, help="SNR(dB) for edge stream"
    )
    parser.add_argument(
        "--image_size", type=int, default=256, help="Resize input to this size"
    )
    parser.add_argument(
        "--steps", type=int, default=500, help="Diffusion inference steps"
    )
    parser.add_argument(
        "--guidance", type=float, default=10, help="CFG guidance scale"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    channel_cfg = ChannelConfig(
        snr_db_text=args.snr_text,
        snr_db_edge=args.snr_edge,
        seed=args.seed,
    )
    gen_cfg = GenerationConfig(
        image_size=args.image_size,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        use_half=True,
    )

    result = run_pipeline(
        input_image_path=args.input,
        output_dir=args.output_dir,
        channel_cfg=channel_cfg,
        gen_cfg=gen_cfg,
    )

    print("\n===== DONE =====")
    print(f"Caption              : {result.caption}")
    print(f"Received caption     : {result.text_packet_received}")
    print(f"Text BER             : {result.text_bit_error_rate:.6f}")
    print(f"Edge BER             : {result.edge_bit_error_rate:.6f}")
    print(f"CLIP score           : {result.clip_score}")
    print(f"MS-SSIM score        : {result.ms_ssim_score}")
    print(f"Edge map path        : {result.edge_map_path}")
    print(f"Output image path    : {result.output_image_path}")
    print(f"Report path          : {result.report_path}")


if __name__ == "__main__":
    main()
