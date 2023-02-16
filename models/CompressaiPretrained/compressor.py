from BaseModel import BaseModel
import argparse

parser = argparse.ArgumentParser(description='Compress image with InvCompress.')
parser.add_argument('arch', metavar='architecture', type=str, help='architecture')
parser.add_argument('img_path', metavar='img path', type=str, help='Img to compress')
parser.add_argument('q', metavar='quality', type=int, help='Quality parametr')
parser.add_argument('save_path', metavar='save path', type=str, help='Save decompressed img')
parser.add_argument('new_img_name', metavar='new name', type=str, help='Save decompressed img')
parser.add_argument('metric', metavar='metric', type=str, help='metric')
parser.add_argument('--cuda', action='store_true', help='use cuda', default=False)

args = parser.parse_args()
path, bpp = (BaseModel(args.arch, args.metric).compress(args.img_path, args.q, args.save_path, args.new_img_name, cuda=args.cuda))
print(path)
print(bpp)