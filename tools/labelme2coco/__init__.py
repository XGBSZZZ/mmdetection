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
    # 提取数据
    coco = get_coco_from_labelme_folder(
        labelme_folder, skip_labels=skip_labels)

    # 保存数据 分割 保存图片
    all_images = {
        Path(i).name: i
        for i in glob(labelme_folder + "/**/*") if Path(i).suffix in
        {'.bmp', '.gif', '.jpeg', '.jpg', '.pbm', '.png', '.tif', '.tiff'}
    }

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
