import json
import json
import os
import erqa
from piqa import LPIPS, FSIM
from piq import vif_p, psnr
from IQA_pytorch import NLPD
from DISTS_pytorch import DISTS
import cv2
from pytorch_msssim import ms_ssim
from torchvision import transforms
from PIL import Image


def _compute_psnr(a, b, max_val: float = 255.0, **kwargs) -> float:
    return psnr(a, b, data_range=max_val).item()


def _compute_ms_ssim(a, b, max_val: float = 255.0, **kwargs) -> float:
    return ms_ssim(a, b, data_range=max_val).item()


def _compute_fsim(a, b, max_val: float = 255.0, **kwargs) -> float:
    fsim = FSIM()
    return fsim(a / max_val, b / max_val).item()


def _compute_lpips(a, b, max_val: float = 255.0, **kwargs) -> float:
    lpips = LPIPS()
    return lpips(a / max_val, b / max_val).item()


def _compute_dists(a, b, max_val: float = 255.0, **kwargs) -> float:
    D = DISTS()
    return D(a / max_val, b / max_val, require_grad=True, batch_average=True).item()


def _compute_erqa(a, b, max_val: float = 255.0, **kwargs) -> float:
    input_path = kwargs['ref_img_path']
    orig = kwargs['decompressed_img_path']
    target = cv2.imread(input_path)
    gt = cv2.imread(orig)
    erqa_metric = erqa.ERQA()
    return erqa_metric(gt, target)


def _compute_vifp(a, b, max_val: float = 255.0, **kwargs) -> float:
    return vif_p(a / max_val, b / max_val, data_range=1.0, reduction='none').item()


def _compute_nlpd(a, b, max_val: float = 255.0, **kwargs) -> float:
    nlpd = NLPD()
    return nlpd(a / max_val, b / max_val).item()


def _compute_vmaf(ref, dec, max_val: float = 255.0, **kwargs):
    pid = os.getpid()
    input_path = kwargs['ref_img_path']
    orig = kwargs['decompressed_img_path']
    os.system(
        f"vqmt -metr vmaf -orig {orig} -in {input_path} -json-file ./vmaf{pid}.json >/dev/null")

    with open(f'./vmaf{pid}.json', 'r') as compJson:
        data = json.load(compJson)
    os.remove(f"./vmaf{pid}.json")

    return data["values"][0]["data"]['A']


def _compute_butteraugli(a, b, max_val: float = 255.0, **kwargs) -> float:
    input_path = kwargs['ref_img_path']
    orig = kwargs['decompressed_img_path']
    score = float(os.popen(f"butteraugli {orig} {input_path}").read())
    return score


_metric_functions = {
    "PSNR": _compute_psnr,
    "MS-SSIM": _compute_ms_ssim,
    "FSIM": _compute_fsim,
    "LPIPS": _compute_lpips,
    "DISTS": _compute_dists,
    "ERQA": _compute_erqa,
    "VIF(P)": _compute_vifp,
    "NLPD": _compute_nlpd,
    "VMAF": _compute_vmaf,
    "butteraugli": _compute_butteraugli,

}


def compute_metrics(ref_img_path, decompressed_img_path, computed_metrics=None):
    if not computed_metrics:
        computed_metrics = []

    ref_img = transforms.ToTensor()(Image.open(ref_img_path).convert("RGB")).unsqueeze(0) * 255
    decompressed_img = transforms.ToTensor()(Image.open(decompressed_img_path).convert("RGB")).unsqueeze(0) * 255
    metrics_values = dict((name, func(ref_img, decompressed_img, ref_img_path=ref_img_path, decompressed_img_path=decompressed_img_path))
                          for name, func in _metric_functions.items() if name not in computed_metrics)
    return metrics_values
