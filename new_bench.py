import argparse
from ast import arg
from time import time
from my_codecs import CompressAIPretrainedCodecs, methods
import os
from tqdm import tqdm
from joblib import Parallel, delayed


from utils import Default_Json_Path, Default_Save_Path, get_bpp, is_present, move_img, move_json, no_ext, get_default_img_path, get_default_json_path, collect_images,  create_json, update_json
from metrics import compute_metrics


def compress_img_compute_metrics(method_name, qps, ref_img_path, save_path, json_path, loss, cuda):
    ref_img_no_ext = no_ext(ref_img_path)
    img_default_save_path = os.path.join(Default_Save_Path, ref_img_no_ext)
    
    if method_name in CompressAIPretrainedCodecs:
        method_name_loss = method_name + '_' + loss
    else:
        method_name_loss = method_name
    
    method = methods[method_name]
    if not os.path.exists(img_default_save_path):
        os.mkdir(img_default_save_path)

    for q in qps:
        #compression
        if not is_present(ref_img_no_ext, q, method_name_loss):
            print(ref_img_path)
            decompressed_path, bpp = method(method_name, ref_img_path, q, img_default_save_path, f'{method_name_loss}_qp={q}', loss=loss, cuda=cuda)
        else:
            decompressed_path = get_default_img_path(ref_img_no_ext, q, method_name_loss)
            bpp = get_bpp(get_default_json_path(ref_img_no_ext), q, method_name_loss)

        #metrics
        metric_val = compute_metrics(ref_img_path, decompressed_path)
        if not os.path.exists(os.path.join(Default_Json_Path, f'{ref_img_no_ext}.json')):
            create_json(ref_img_no_ext, method_name_loss, q, bpp, metric_val, Default_Json_Path)
        else:
            update_json(ref_img_no_ext, method_name_loss, q, bpp, metric_val, Default_Json_Path)

        #save results
        if save_path:
            move_img(ref_img_no_ext, decompressed_path, save_path, bpp, method_name_loss)
        if json_path:
            move_json(ref_img_no_ext, json_path)


def arg_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d' '--dataset_path', type=str, required=True, help='Img to compress')
    parser.add_argument('-q', '--quality', type=int, nargs='+', required=True, help='Quality parametrs')
    parser.add_argument('-a', '--architecture', type=str, required=True, help='Save decompressed img')
    parser.add_argument('-s', '--save_img_dir', type=str, help='Save decompressed img')
    parser.add_argument('-j', '--save_json_dir', type=str, help='Save jsons')
    parser.add_argument('-l', '--loss', type=str, help='Loss function', default='mse')
    parser.add_argument('--cuda', action='store_true', help='Use cuda', default=False)
    parser.add_argument('--n_jobs', type=int, help='n_jobs', default=1)

    return parser

def main():
    args = arg_parser().parse_args()
    method_name = args.architecture    
    loss = args.loss
    qps = args.quality
    dataset_path = args.d__dataset_path
    save_path = args.save_img_dir
    json_path = args.save_json_dir
    cuda = args.cuda
    n_jobs = args.n_jobs

    imgs = collect_images(dataset_path)

    wrap = lambda img_path: compress_img_compute_metrics(method_name, qps, img_path, save_path, json_path, loss, cuda)

    Parallel(n_jobs=n_jobs)(delayed(wrap)(img_path) for img_path in tqdm(imgs))


if __name__ == '__main__':
    main()