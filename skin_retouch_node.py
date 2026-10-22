"""ComfyUI node wrapper for the lightweight skin retouch core."""

import cv2
import numpy as np
import torch

from . import skin_core


class SkinRetouch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {
                    "default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Overall retouch amount.",
                }),
                "texture_keep": ("FLOAT", {
                    "default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Pore-level texture kept. 0 = plastic, 0.6 = barely retouched.",
                }),
                "blemish": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blemish removal radius. Higher clears larger spots.",
                }),
                "tone_even": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Redness and blotch levelling.",
                }),
                "feather": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Mask edge softness, relative to face size.",
                }),
            },
            "optional": {
                "max_faces": ("INT", {
                    "default": 4, "min": 1, "max": 16, "step": 1,
                    "tooltip": "Largest faces processed; extra faces are ignored.",
                }),
                "face_confidence": ("FLOAT", {
                    "default": 0.35, "min": 0.05, "max": 0.95, "step": 0.05,
                }),
                "include_body": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Also retouch the main person's arms, shoulders, chest and "
                               "hands. Costs a person-segmentation pass.",
                }),
                "body_strength": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Strength for body skin. 0 = follow 'strength'.",
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Inference and filter device. auto picks CUDA when available.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "skin_mask")
    FUNCTION = "run"
    CATEGORY = "image/retouch"

    def run(self, image, strength, texture_keep, blemish, tone_even, feather,
            max_faces=4, face_confidence=0.35, include_body=True,
            body_strength=0.0, device="auto"):
        out_images = []
        out_masks = []
        body_s = None if body_strength <= 0.0 else float(body_strength)
        for i in range(image.shape[0]):
            src = image[i].cpu().numpy()
            alpha = None
            if src.shape[-1] == 1:
                src = np.repeat(src, 3, axis=-1)
            elif src.shape[-1] == 4:
                alpha = src[..., 3:]
                src = src[..., :3]

            bgr = cv2.cvtColor((src * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
            fixed_bgr, mask, _ = skin_core.skin_retouch(
                bgr, strength=strength, texture_keep=texture_keep,
                blemish=blemish, tone_even=tone_even, feather=feather,
                conf=face_confidence, max_faces=max_faces,
                include_body=bool(include_body), body_strength=body_s,
                device=device,
            )
            rgb = cv2.cvtColor(fixed_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            if alpha is not None:
                rgb = np.concatenate([rgb, alpha], axis=-1)
            elif src.ndim == 3 and src.shape[-1] == 3 and image.shape[-1] == 1:
                rgb = rgb[..., :1]
            out_images.append(torch.from_numpy(rgb))
            out_masks.append(torch.from_numpy(mask.astype(np.float32) / 255.0))

        return (torch.stack(out_images), torch.stack(out_masks))


NODE_CLASS_MAPPINGS = {"SkinRetouch": SkinRetouch}
NODE_DISPLAY_NAME_MAPPINGS = {"SkinRetouch": "Skin Retouch (Lite)"}
