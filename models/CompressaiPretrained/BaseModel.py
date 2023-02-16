import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torchvision import transforms
from PIL import Image
from compressai.zoo.image import model_architectures as architectures
from compressai.zoo import models as pretrained_models

def read_image(filepath: str) -> torch.Tensor:
    assert os.path.isfile(filepath)
    img = Image.open(filepath).convert("RGB")

    return transforms.ToTensor()(img)


def load_pretrained(model: str, metric: str, quality: int) -> nn.Module:
    return pretrained_models[model](
        quality=quality, metric=metric, pretrained=True
    ).eval()


class BaseModel:
    def __init__(self, arch, metric) -> None:
        self.metric = metric
        self.arch = arch


    def compress(self, img_path, quality, savePath, new_img_name, cuda=False, half=False):
        model = load_pretrained(self.arch, self.metric, quality)
        if cuda and torch.cuda.is_available():
            model = model.to("cuda")
        device = next(model.parameters()).device
        x = read_image(img_path).to(device)
        if half:
            model = model.half()
            x = x.half()

        x = x.unsqueeze(0)

        h, w = x.size(2), x.size(3)
        p = 64  # maximum 6 strides of 2
        new_h = (h + p - 1) // p * p
        new_w = (w + p - 1) // p * p
        padding_left = (new_w - w) // 2
        padding_right = new_w - w - padding_left
        padding_top = (new_h - h) // 2
        padding_bottom = new_h - h - padding_top
        x_padded = F.pad(
            x,
            (padding_left, padding_right, padding_top, padding_bottom),
            mode="constant",
            value=0,
        )
        start = time.time()
        out_enc = model.compress(x_padded)
        enc_time = time.time() - start

        start = time.time()
        out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
        dec_time = time.time() - start
        out_dec["x_hat"] = F.pad(
            out_dec["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
        )

        num_pixels = x.size(0) * x.size(2) * x.size(3)
        bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
        assert savePath, 'Invalid save path'
        if not os.path.exists(savePath):
            os.makedirs(savePath)

        tran1 = transforms.ToPILImage()

        cur_img = tran1(out_dec["x_hat"][0])
        decompressedImgPath = os.path.join(savePath, f'{new_img_name}.png')
        cur_img.save(decompressedImgPath)
        return decompressedImgPath, bpp

