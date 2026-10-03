import open_clip
import torch
import torch.nn as nn
from contextlib import nullcontext
from peft import LoraConfig, get_peft_model


class VisionEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        clip, _, self.preprocess = open_clip.create_model_and_transforms(args.encoder_model)
        self.visual_encoder = clip.visual

        for p in self.visual_encoder.parameters():
            p.requires_grad = False

        self.use_lora = getattr(args, "use_lora", True)
        if self.use_lora:
            config = LoraConfig(
                r=getattr(args, "lora_r", 16),
                lora_alpha=getattr(args, "lora_alpha", 32),
                lora_dropout=getattr(args, "lora_dropout", 0.05),
                target_modules=r"transformer\.resblocks\.\d+\.(attn\.out_proj|mlp\.c_fc|mlp\.c_proj)",
                bias="none",
            )
            self.visual_encoder = get_peft_model(self.visual_encoder, config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        ctx = nullcontext() if self.use_lora else torch.no_grad()
        with ctx:
            features = self.visual_encoder.forward_intermediates(pixel_values)
            image_features = features["image_intermediates"][11]
            image_features = image_features.reshape(
                image_features.shape[0], image_features.shape[1], -1
            )
        return image_features