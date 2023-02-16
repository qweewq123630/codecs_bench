import os
import shutil
import json

ROOT_DIR = '/home/vitaliy/codecs_bench'
Default_Save_Path = '/home/vitaliy/codecs_bench/Imgs/AllCompressed'
Default_Json_Path = '/home/vitaliy/codecs_bench/Jsons/AllJsons'


IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)

precision = 1000


def collect_images(rootpath: str):
    return sorted([
        os.path.join(rootpath, f)
        for f in os.listdir(rootpath)
        if os.path.splitext(f)[-1].lower() in IMG_EXTENSIONS
    ])


def get_default_img_path(img_name, q, method_name):
    return os.path.join(Default_Save_Path, img_name, f'{method_name}_qp={q}.png')


def get_default_json_path(img_name):
    return os.path.join(Default_Json_Path, f'{img_name}.json')


def get_bpp(json_path, q, method_name):
    with open(json_path, 'r') as openfile:
        json_object = json.load(openfile)
    return json_object[method_name][str(q)]['bpp']


def is_present(img, q, method_name):
    if not os.path.exists(get_default_img_path(img, q, method_name)):
        return False
    json_path = get_default_json_path(img)
    if not os.path.exists(json_path):
        return False
    with open(json_path, 'r') as openfile:
        json_object = json.load(openfile)
    if method_name not in json_object.keys():
        return False
    return str(q) in json_object[method_name].keys()


def move_img(ref_img, decompressed_path, save_path, bpp, method_name):
    dir_name = os.path.join(save_path, ref_img)
    if not os.path.exists(dir_name):
        os.mkdir(os.path.join(save_path, ref_img))
    shutil.copy(decompressed_path, os.path.join(save_path, ref_img, img_save_name_with_bpp(method_name, bpp)))


def move_json(img, new_json_path):
    shutil.copy(get_default_json_path(img), new_json_path)


def no_ext(name):
    return name[name.rfind('/') + 1:name.rfind('.')]


def img_save_name_with_bpp(method_name, bpp):
    return f'{method_name}@bpp={int(bpp * precision)}.png'


def img_save_path_with_bpp_qp(path, method_name, bpp, qp):
    return os.path.join(path, f'{method_name}_@qp={qp}_@bpp={int(bpp * precision)}.png')


def create_json(img_name, method_name, q, bpp, metric_values, save_path):
    res = {method_name: {
        q: {
            'bpp': bpp,
            'metrics': metric_values
        }
    }}
    json_path = os.path.join(save_path, f'{img_name}.json')
    with open(json_path, "w") as outfile:
        json.dump(res, outfile, indent=2)


def update_json(img_name, method_name, q, bpp, metric_values, save_path):
    json_path = os.path.join(save_path, f'{img_name}.json')
    
    with open(json_path, 'r') as openfile:
        json_object = json.load(openfile)

    if method_name in json_object.keys():
        json_object[method_name].update({
            str(q): {
                'bpp': bpp,
                'metrics': metric_values
            }
        })
    else:
        json_object.update(
            {method_name: {
                str(q): {
                    'bpp': bpp,
                    'metrics': metric_values
                }}})
    with open(json_path, "w") as outfile:
        json.dump(json_object, outfile, indent=2)
