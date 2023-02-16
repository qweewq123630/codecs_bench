import argparse
from .Codecs import AV1, BPG, HM, JPEG, JPEG2000, TFCI, VTM, WebP

codecs = [JPEG, WebP, JPEG2000, BPG, TFCI, VTM, HM, AV1]    


parser = argparse.ArgumentParser(description='')
parser.add_argument('architecture', metavar='architecture', type=str, help='architecture')
parser.add_argument('img_path', metavar='img path', type=str, help='Img to compress')
parser.add_argument('q', metavar='quality', type=int, help='Quality parametr')
parser.add_argument('save_path', metavar='save path', type=str, help='Save decompressed img')
parser.add_argument('new_img_name', metavar='new name', type=str, help='Save decompressed img')

def traditional_compress(arch, img_path, q, save_path, new_img_name):
    codec_cls = next(c for c in codecs if c.__name__.lower() == arch)
    codec = codec_cls(dict())
    path, bpp = (codec.run(img_path, q, save_path, new_img_name))
    return path, bpp