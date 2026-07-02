"""Device-agnostic loader and inference wrapper for DSINE (Bae & Davison, CVPR 2024).

Uses the vendored, CPU/MPS-compatible copy of the model architecture in
external/dsine/ (adapted from https://github.com/baegwangbin/DSINE, which
hardcodes CUDA). No training happens here: we instantiate the published
architecture and load the publicly released checkpoint (inference only).
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

VENDOR_DIR = Path(__file__).resolve().parent.parent / "external" / "dsine"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from models.dsine.v02 import DSINE_v02  # noqa: E402

_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def _dsine_args():
    # Matches DSINE-main/projects/dsine/experiments/exp001_cvpr2024/dsine.txt,
    # the config used to produce the released checkpoint.
    return SimpleNamespace(
        NNET_architecture="v02",
        NNET_encoder_B=5,
        NNET_output_dim=3,
        NNET_output_type="R",
        NNET_feature_dim=64,
        NNET_hidden_dim=64,
        NNET_decoder_NF=2048,
        NNET_decoder_BN=False,
        NNET_decoder_down=8,
        NNET_learned_upsampling=True,
        NRN_prop_ps=5,
        NRN_num_iter_train=5,
        NRN_num_iter_test=5,
        NRN_ray_relu=True,
    )


def _get_padding(orig_h, orig_w):
    """Pad so H and W are multiples of 32 (required by the encoder's 5x downsampling)."""
    if orig_w % 32 == 0:
        l = r = 0
    else:
        new_w = 32 * ((orig_w // 32) + 1)
        l = (new_w - orig_w) // 2
        r = (new_w - orig_w) - l
    if orig_h % 32 == 0:
        t = b = 0
    else:
        new_h = 32 * ((orig_h // 32) + 1)
        t = (new_h - orig_h) // 2
        b = (new_h - orig_h) - t
    return l, r, t, b


def _intrins_from_fov(fov_deg, h, w, device):
    """Camera intrinsics assuming centered principal point and a fixed field-of-view.

    DSINE conditions on per-pixel ray direction, which requires camera intrinsics.
    We don't have real calibration for arbitrary test photos, so we fall back to the
    same assumption used in the official demo scripts (60-degree FOV, centered).
    """
    f = (max(h, w) / 2.0) / np.tan(np.deg2rad(fov_deg / 2.0))
    cx, cy = (w / 2.0) - 0.5, (h / 2.0) - 0.5
    return torch.tensor([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=torch.float32, device=device)


def load_dsine(checkpoint_path, device):
    """Instantiate DSINE_v02 and load the pretrained checkpoint onto `device`."""
    model = DSINE_v02(_dsine_args())
    ckpt = torch.load(checkpoint_path, map_location="cpu")["model"]
    state_dict = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.pixel_coords = model.pixel_coords.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_normals(model, image: Image.Image, device, fov_deg=60.0):
    """Run DSINE on a PIL RGB image. Returns an (H, W, 3) float32 array in [-1, 1]."""
    img = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    _, _, orig_h, orig_w = img_t.shape
    l, r, t, b = _get_padding(orig_h, orig_w)
    img_t = F.pad(img_t, (l, r, t, b), mode="constant", value=0.0)
    img_t = _NORMALIZE(img_t)

    intrins = _intrins_from_fov(fov_deg, orig_h, orig_w, device).unsqueeze(0)
    intrins[:, 0, 2] += l
    intrins[:, 1, 2] += t

    pred_norm = model(img_t, intrins=intrins, mode="test")[-1]
    pred_norm = pred_norm[:, :, t:t + orig_h, l:l + orig_w]
    return pred_norm[0].permute(1, 2, 0).cpu().numpy()
