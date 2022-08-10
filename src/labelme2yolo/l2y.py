"""
Created on Aug 18, 2021

@author: xiaosonh
@author: GreatV(Wang Xin)
"""
import base64
import io
import json
import math
import os
import shutil
from collections import OrderedDict
from multiprocessing import Pool

import cv2
import numpy as np
import PIL.ExifTags
import PIL.Image
import PIL.ImageOps
from sklearn.model_selection import train_test_split


# copy form https://github.com/wkentaro/labelme/blob/main/labelme/utils/image.py
def img_data_to_pil(img_data):
    f = io.BytesIO()
    f.write(img_data)
    img_pil = PIL.Image.open(f)
    return img_pil


# copy form https://github.com/wkentaro/labelme/blob/main/labelme/utils/image.py
def img_data_to_arr(img_data):
    img_pil = img_data_to_pil(img_data)
    img_arr = np.array(img_pil)
    return img_arr


# copy form https://github.com/wkentaro/labelme/blob/main/labelme/utils/image.py
def img_b64_to_arr(img_b64):
    img_data = base64.b64decode(img_b64)
    img_arr = img_data_to_arr(img_data)
    return img_arr


# copy form https://github.com/wkentaro/labelme/blob/main/labelme/utils/image.py
def img_pil_to_data(img_pil):
    f = io.BytesIO()
    img_pil.save(f, format="PNG")
    img_data = f.getvalue()
    return img_data


# copy form https://github.com/wkentaro/labelme/blob/main/labelme/utils/image.py
def img_arr_to_b64(img_arr):
    img_pil = PIL.Image.fromarray(img_arr)
    f = io.BytesIO()
    img_pil.save(f, format="PNG")
    img_bin = f.getvalue()
    if hasattr(base64, "encodebytes"):
        img_b64 = base64.encodebytes(img_bin)
    else:
        img_b64 = base64.encodestring(img_bin)
    return img_b64


# copy form https://github.com/wkentaro/labelme/blob/main/labelme/utils/image.py
def img_data_to_png_data(img_data):
    with io.BytesIO() as f_out:
        f_out.write(img_data)
        img = PIL.Image.open(f_out)

        with io.BytesIO() as f_in:
            img.save(f_in, "PNG")
            f_in.seek(0)
            return f_in.read()


def get_label_id_map(json_dir):
    label_set = set()

    for file_name in os.listdir(json_dir):
        if file_name.endswith("json"):
            json_path = os.path.join(json_dir, file_name)
            data = json.load(open(json_path))
            for shape in data["shapes"]:
                label_set.add(shape["label"])

    return OrderedDict([(label, label_id) for label_id, label in enumerate(label_set)])


def save_yolo_label(json_name, label_dir_path, target_dir, yolo_obj_list):
    txt_path = os.path.join(
        label_dir_path, target_dir, json_name.replace(".json", ".txt")
    )

    with open(txt_path, "w+") as f:
        for yolo_obj_idx, yolo_obj in enumerate(yolo_obj_list):
            yolo_obj_line = (
                "%s %s %s %s %s\n" % yolo_obj
                if yolo_obj_idx + 1 != len(yolo_obj_list)
                else "%s %s %s %s %s" % yolo_obj
            )
            f.write(yolo_obj_line)


def save_yolo_image(json_data, json_name, image_dir_path, target_dir):
    img_name = json_name.replace(".json", ".png")
    img_path = os.path.join(image_dir_path, target_dir, img_name)

    if not os.path.exists(img_path):
        img = img_b64_to_arr(json_data["imageData"])
        PIL.Image.fromarray(img).save(img_path)

    return img_path


class Labelme2YOLO(object):
    def __init__(self, json_dir):
        self._json_dir = json_dir

        self._label_id_map = get_label_id_map(self._json_dir)

    def _make_train_val_dir(self):
        self._label_dir_path = os.path.join(self._json_dir, "YOLODataset/labels/")
        self._image_dir_path = os.path.join(self._json_dir, "YOLODataset/images/")

        for yolo_path in (
            os.path.join(self._label_dir_path + "train/"),
            os.path.join(self._label_dir_path + "val/"),
            os.path.join(self._label_dir_path + "test/"),
            os.path.join(self._image_dir_path + "train/"),
            os.path.join(self._image_dir_path + "val/"),
            os.path.join(self._image_dir_path + "test/"),
        ):
            if os.path.exists(yolo_path):
                shutil.rmtree(yolo_path)

            os.makedirs(yolo_path)

    def _train_test_split(self, folders, json_names, val_size, test_size):
        if (
            len(folders) > 0
            and "train" in folders
            and "val" in folders
            and "test" in folders
        ):
            train_json_names = self.get_json_names("train/")
            val_json_names = self.get_json_names("val/")
            test_json_names = self.get_json_names("test/")

            return train_json_names, val_json_names, test_json_names

        train_indexes, val_indexes = train_test_split(
            range(len(json_names)), test_size=val_size
        )
        tmp_train_len = len(train_indexes)
        test_indexes = []
        if test_size:
            train_indexes, test_indexes = train_test_split(
                range(tmp_train_len), test_size=test_size / (1 - val_size)
            )
        train_json_names = [json_names[train_idx] for train_idx in train_indexes]
        val_json_names = [json_names[val_idx] for val_idx in val_indexes]
        test_json_names = [json_names[test_idx] for test_idx in test_indexes]

        return train_json_names, val_json_names, test_json_names

    def get_json_names(self, data_type: str):
        data_folder = os.path.join(self._json_dir, data_type)
        data_json_names = [
            data_sample_name + ".json"
            for data_sample_name in os.listdir(data_folder)
            if os.path.isdir(os.path.join(data_folder, data_sample_name))
        ]
        return data_json_names

    def convert(self, val_size, test_size):
        json_names = [
            file_name
            for file_name in os.listdir(self._json_dir)
            if os.path.isfile(os.path.join(self._json_dir, file_name))
            and file_name.endswith(".json")
        ]
        folders = [
            file_name
            for file_name in os.listdir(self._json_dir)
            if os.path.isdir(os.path.join(self._json_dir, file_name))
        ]
        train_json_names, val_json_names, test_json_names = self._train_test_split(
            folders, json_names, val_size, test_size
        )

        self._make_train_val_dir()

        # convert labelme object to yolo format object, and save them to files
        # also get image from labelme json file and save them under images folder
        for target_dir, json_names in zip(
            ("train/", "val/", "test/"),
            (train_json_names, val_json_names, test_json_names),
        ):
            pool = Pool(os.cpu_count() - 1)
            for json_name in json_names:
                pool.apply_async(self.covert_json_to_text, args=(target_dir, json_name))
            pool.close()
            pool.join()

        print("Generating dataset.yaml file ...")
        self._save_dataset_yaml()

    def covert_json_to_text(self, target_dir, json_name):
        json_path = os.path.join(self._json_dir, json_name)
        json_data = json.load(open(json_path))

        print("Converting %s for %s ..." % (json_name, target_dir.replace("/", "")))

        img_path = save_yolo_image(
            json_data, json_name, self._image_dir_path, target_dir
        )

        yolo_obj_list = self._get_yolo_object_list(json_data, img_path)
        save_yolo_label(json_name, self._label_dir_path, target_dir, yolo_obj_list)

    def convert_one(self, json_name):
        json_path = os.path.join(self._json_dir, json_name)
        json_data = json.load(open(json_path))

        print("Converting %s ..." % json_name)

        img_path = save_yolo_image(json_data, json_name, self._json_dir, "")

        yolo_obj_list = self._get_yolo_object_list(json_data, img_path)
        save_yolo_label(json_name, self._json_dir, "", yolo_obj_list)

    def _get_yolo_object_list(self, json_data, img_path):
        yolo_obj_list = []

        img_h, img_w, _ = cv2.imread(img_path).shape
        for shape in json_data["shapes"]:
            # labelme circle shape is different from others
            # it only has 2 points, 1st is circle center, 2nd is drag end point
            if shape["shape_type"] == "circle":
                yolo_obj = self._get_circle_shape_yolo_object(shape, img_h, img_w)
            else:
                yolo_obj = self._get_other_shape_yolo_object(shape, img_h, img_w)

            yolo_obj_list.append(yolo_obj)

        return yolo_obj_list

    def _get_circle_shape_yolo_object(self, shape, img_h, img_w):
        obj_center_x, obj_center_y = shape["points"][0]

        radius = math.sqrt(
            (obj_center_x - shape["points"][1][0]) ** 2
            + (obj_center_y - shape["points"][1][1]) ** 2
        )
        obj_w = 2 * radius
        obj_h = 2 * radius

        yolo_center_x = round(float(obj_center_x / img_w), 6)
        yolo_center_y = round(float(obj_center_y / img_h), 6)
        yolo_w = round(float(obj_w / img_w), 6)
        yolo_h = round(float(obj_h / img_h), 6)

        label_id = self._label_id_map[shape["label"]]

        return label_id, yolo_center_x, yolo_center_y, yolo_w, yolo_h

    def _get_other_shape_yolo_object(self, shape, img_h, img_w):
        def __get_object_desc(obj_port_list):
            def __get_dist(int_list):
                return max(int_list) - min(int_list)

            x_lists = [port[0] for port in obj_port_list]
            y_lists = [port[1] for port in obj_port_list]

            return min(x_lists), __get_dist(x_lists), min(y_lists), __get_dist(y_lists)

        obj_x_min, obj_w, obj_y_min, obj_h = __get_object_desc(shape["points"])

        yolo_center_x = round(float((obj_x_min + obj_w / 2.0) / img_w), 6)
        yolo_center_y = round(float((obj_y_min + obj_h / 2.0) / img_h), 6)
        yolo_w = round(float(obj_w / img_w), 6)
        yolo_h = round(float(obj_h / img_h), 6)

        label_id = self._label_id_map[shape["label"]]

        return label_id, yolo_center_x, yolo_center_y, yolo_w, yolo_h

    def _save_dataset_yaml(self):
        yaml_path = os.path.join(self._json_dir, "YOLODataset", "dataset.yaml")

        with open(yaml_path, "w+") as yaml_file:
            yaml_file.write("train: %s\n" % os.path.join(self._image_dir_path, "train"))
            yaml_file.write("val: %s\n\n" % os.path.join(self._image_dir_path, "val"))
            yaml_file.write("test: %s\n\n" % os.path.join(self._image_dir_path, "test"))
            yaml_file.write("nc: %i\n\n" % len(self._label_id_map))

            names_str = ""
            for label, _ in self._label_id_map.items():
                names_str += "'%s', " % label
            names_str = names_str.rstrip(", ")
            yaml_file.write("names: [%s]" % names_str)
