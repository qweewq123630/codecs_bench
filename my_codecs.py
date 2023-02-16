import os
from utils import ROOT_DIR
from models.CompressaiPretrained.BaseModel import BaseModel as PretrainedCompressAIModel
from models.CompressaiTraditional.compressor import traditional_compress 
from models.InvCompress.InvCompress import InvCompress as InvCompressClass
from models.Coarse2Fine.Coarse2Fine import compress_coarse2fine
import torch

SCRIPT_DIR = './run_scripts'
CompressAIPretrainedCodecs = ['cheng2020-anchor',
    'mbt2018-mean',
    'bmshj2018-factorized',
    'bmshj2018-hyperprior',
    'mbt2018',
    'cheng2020-attn',
    'mbt2018-mean',
    'coarse2fine']


@torch.no_grad()
def InvCompress(arch, img_path, q, save_path, new_img_name, **kwargs):
    compressor = InvCompressClass('./models/InvCompress/experiments') 
    path, bpp = compressor.compress(img_path, q, save_path, new_img_name, cuda=kwargs['cuda'])

    return path, float(bpp)


@torch.no_grad()
def CompressaiPretrained(arch, img_path, q, save_path, new_img_name, **kwargs):
    path, bpp = PretrainedCompressAIModel(arch, kwargs['loss']).compress(img_path, q, save_path, new_img_name, cuda=kwargs['cuda'])
    return path, float(bpp)


@torch.no_grad()
def CompressaiTraditional(arch, img_path, q, save_path, new_img_name, **kwargs):
    path, bpp = traditional_compress(arch, img_path, q, save_path, new_img_name)
    return path, float(bpp)


@torch.no_grad()
def Coarse2Fine(arch, img_path, q, save_path, new_img_name, **kwargs):
    loss = kwargs['loss']
    device = 'cuda' if kwargs['cuda'] else 'cpu'
    path, bpp = compress_coarse2fine(img_path, save_path, q, new_img_name, loss, device)
    return path, float(bpp)


methods = {
    'cheng2020-anchor': CompressaiPretrained,
    'mbt2018-mean': CompressaiPretrained,
    'bmshj2018-factorized': CompressaiPretrained,
    'bmshj2018-hyperprior': CompressaiPretrained,
    'mbt2018': CompressaiPretrained,
    'cheng2020-attn': CompressaiPretrained,
    'mbt2018-mean': CompressaiPretrained,

    'hm': CompressaiTraditional,
    'vtm': CompressaiTraditional,
    'jpeg': CompressaiTraditional,
    'jpeg2000': CompressaiTraditional,
    'webp': CompressaiTraditional,
    'bpg': CompressaiTraditional,
    'jpeg': CompressaiTraditional,

    'invcompress': InvCompress,
    'coarse2fine': Coarse2Fine
}
