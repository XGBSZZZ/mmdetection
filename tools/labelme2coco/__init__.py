from __future__ import absolute_import

__version__ = "0.2.4"

import logging
import os
from pathlib import Path
from glob import glob
from multiprocessing.dummy import Pool
from sahi.utils.file import save_json
import shutil
from tools.labelme2coco.labelme2coco import get_coco_from_labelme_folder
from sahi.utils.file import list_files_recursively, load_json
from typing import List

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
)


def convert(
    labelme_folder: str,
    export_dir: str = "runs/labelme2coco/",
    train_split_rate: float = 1,
    skip_labels: List[str] = [],
):
    # 权限
    import paramiko
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('eight', username='tangfengwei', password='qweasdzxc')

    # 更改文件权限
    stdin, stdout, stderr = ssh.exec_command(f'chmod +777 {export_dir}')
    print(stdout.read().decode())
    ssh.close()
    """
    Args:
        labelme_folder: folder that contains labelme annotations and image files
        export_dir: path for coco jsons to be exported
        train_split_rate: ration fo train split
    """
    # 删除垃圾数据
    all_images = {
        Path(i).name: i
        for i in glob(labelme_folder + "/**/*", recursive=True) if Path(i).suffix in
        {'.bmp', '.gif', '.jpeg', '.jpg', '.pbm', '.png', '.tif', '.tiff'}
    }

    bad_dir = str(Path(labelme_folder).parent) + "/bad"
    fuck_numb = 0
    if not os.path.exists(bad_dir):
        os.makedirs(bad_dir)
    _, abs_json_path_list = list_files_recursively(labelme_folder, contains=[".json"])
    try:
        for json_path in abs_json_path_list:
            if load_json(json_path)["imagePath"] not in all_images.keys():
                print("错误的JSON: ", load_json(json_path)["imagePath"])
                os.remove(json_path)
                fuck_numb += 1
    except:
        # shutil.move(json_path, bad_dir)
        os.remove(json_path)
        fuck_numb += 1
    logger.critical(f"垃圾JSON: {fuck_numb}")

    # 提取数据
    coco = get_coco_from_labelme_folder(
        labelme_folder, skip_labels=skip_labels)

    logger.info([i["name"] for i in coco.json["categories"]])

    # 保存数据 分割 保存图片

    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    os.makedirs(export_dir)

    def save_images(images_path, name):
        name = '/' + name
        if os.path.exists(export_dir + name):
            shutil.rmtree(export_dir + name)
        os.makedirs(export_dir + name)

        def save_image(image_path):
            shutil.copy(image_path, export_dir + name)
            return True

        with Pool(processes=8) as pool:
            ret = pool.map_async(func=save_image, iterable=images_path)
            ret.wait()

    if train_split_rate < 1:
        result = coco.split_coco_as_train_val(train_split_rate)
        # export train split
        save_path = str(Path(export_dir) / "train.json")
        save_json(result["train_coco"].json, save_path)
        logger.info(
            f"Training split in COCO format is exported to {save_path}")

        # export val split
        save_path = str(Path(export_dir) / "val.json")
        save_json(result["val_coco"].json, save_path)
        logger.info(
            f"Validation split in COCO format is exported to {save_path}")

        save_images([all_images[i["file_name"]] for i in result["train_coco"].json["images"]], "train")
        save_images([all_images[i["file_name"]] for i in result["val_coco"].json["images"]], "val")
    else:
        save_path = str(Path(export_dir) / "dataset.json")
        save_json(coco.json, save_path)
        logger.info(
            f"Converted annotations in COCO format is exported to {save_path}")

        images = [all_images[i["file_name"]] for i in coco.json.json["images"]]
        save_images(images, "img")

    logging.info("都结束啦")
    logging.info("都结束啦")
    logging.info("都结束啦")
