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


# number of LabelMe2YOLO multiprocessing threads
NUM_THREADS = max(1, os.cpu_count() - 1)


# copy form https://github.com/wkentaro/labelme/blob/main/labelme/utils/image.py
def img_data_to_pil(img_data):
    '''Convert img_data(byte) to PIL.Image'''
    file = io.BytesIO()
    file.write(img_data)
    img_pil = PIL.Image.open(file)
    return img_pil


# copy form https://github.com/wkentaro/labelme/blob/main/labelme/utils/image.py
def img_data_to_arr(img_data):
    '''Convert img_data(byte) to numpy.ndarray'''
    img_pil = img_data_to_pil(img_data)
    img_arr = np.array(img_pil)
    return img_arr


# copy form https://github.com/wkentaro/labelme/blob/main/labelme/utils/image.py
def img_b64_to_arr(img_b64):
    '''Convert img_b64(str) to numpy.ndarray'''
    img_data = base64.b64decode(img_b64)
    img_arr = img_data_to_arr(img_data)
    return img_arr


# copy form https://github.com/wkentaro/labelme/blob/main/labelme/utils/image.py
def img_pil_to_data(img_pil):
    '''Convert PIL.Image to img_data(byte)'''
    file = io.BytesIO()
    img_pil.save(file, format="PNG")
    img_data = file.getvalue()
    return img_data


# copy form https://github.com/wkentaro/labelme/blob/main/labelme/utils/image.py
def img_arr_to_b64(img_arr):
    '''Convert numpy.ndarray to img_b64(str)'''
    img_pil = PIL.Image.fromarray(img_arr)
    file = io.BytesIO()
    img_pil.save(file, format="PNG")
    img_bin = file.getvalue()
    if hasattr(base64, "encodebytes"):
        img_b64 = base64.encodebytes(img_bin)
    else:
        img_b64 = base64.encodestring(img_bin)
    return img_b64


# copy form https://github.com/wkentaro/labelme/blob/main/labelme/utils/image.py
def img_data_to_png_data(img_data):
    '''Convert img_data(byte) to png_data(byte)'''
    with io.BytesIO() as f_out:
        f_out.write(img_data)
        img = PIL.Image.open(f_out)

        with io.BytesIO() as f_in:
            img.save(f_in, "PNG")
            f_in.seek(0)
            return f_in.read()


def get_label_id_map(json_dir: str):
    '''Get label id map from json files in json_dir'''
    label_set = set()

    for file_name in os.listdir(json_dir):
        if file_name.endswith("json"):
            json_path = os.path.join(json_dir, file_name)
            data = json.load(open(json_path))
            for shape in data["shapes"]:
                label_set.add(shape["label"])

    return OrderedDict([(label, label_id) for label_id, label in enumerate(label_set)])


def extend_point_list(point_list, out_format="polygon"):
    '''Extend point list to polygon or bbox'''
    xmin = min([float(point) for point in point_list[::2]])
    xmax = max([float(point) for point in point_list[::2]])
    ymin = min([float(point) for point in point_list[1::2]])
    ymax = max([float(point) for point in point_list[1::2]])

    if out_format == "polygon":
        return np.array([xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax])
    if out_format == "bbox":
        x = xmin
        y = ymin
        w = xmax - xmin
        h = ymax - ymin
        x = x + w / 2
        y = y + h / 2
        return np.array([x, y, w, h])


def save_yolo_label(json_name, label_dir_path, target_dir, yolo_obj_list):
    '''Save yolo label to txt file'''
    txt_path = os.path.join(label_dir_path,
                            target_dir,
                            json_name.replace(".json", ".txt"))

    with open(txt_path, "w+") as f:
        for yolo_obj in yolo_obj_list:
            label, points = yolo_obj
            points = [str(item) for item in points]
            yolo_obj_line = f"{label} {' '.join(points)}\n"
            f.write(yolo_obj_line)


def save_yolo_image(json_data, json_path, image_dir_path, target_dir):
    '''Save yolo image to image_dir_path/target_dir'''
    json_name = os.path.basename(json_path)
    img_name = json_name.replace(".json", ".png")

    # make image_path and save image
    img_path = os.path.join(image_dir_path, target_dir, img_name)

    if json_data["imageData"] is None:
        dirname = os.path.dirname(json_path)
        image_name = json_data["imagePath"]
        src_image_name = os.path.join(dirname, image_name)
        src_image = cv2.imread(src_image_name)
        cv2.imwrite(img_path, src_image)
    else:
        img = img_b64_to_arr(json_data["imageData"])
        PIL.Image.fromarray(img).save(img_path)

    return img_path


class Labelme2YOLO(object):
    '''Labelme to YOLO format converter'''
    def __init__(self, json_dir, output_format, label_list):
        self._json_dir = json_dir
        self._output_format = output_format
        self._label_list = label_list

        if label_list:
            self._label_id_map = {label: label_id
                                  for label_id, label in enumerate(label_list)}
        else:
            self._label_id_map = get_label_id_map(self._json_dir)
            self._label_list = [label for label in self._label_id_map.keys()]

    def _make_train_val_dir(self):
        self._label_dir_path = os.path.join(self._json_dir,
                                            'YOLODataset/labels/')
        self._image_dir_path = os.path.join(self._json_dir,
                                            'YOLODataset/images/')

        for yolo_path in (os.path.join(self._label_dir_path + 'train/'),
                          os.path.join(self._label_dir_path + 'val/'),
                          os.path.join(self._label_dir_path + 'test/'),
                          os.path.join(self._image_dir_path + 'train/'),
                          os.path.join(self._image_dir_path + 'val/'),
                          os.path.join(self._image_dir_path + 'test/')):
            if os.path.exists(yolo_path):
                shutil.rmtree(yolo_path)

            os.makedirs(yolo_path)

    def _train_test_split(self, folders, json_names, val_size, test_size):
        if len(folders) > 0 and 'train' in folders and 'val' in folders and 'test' in folders:  # noqa: E501
            train_folder = os.path.join(self._json_dir, 'train/')
            train_json_names = [train_sample_name + '.json'
                                for train_sample_name in os.listdir(train_folder)
                                if os.path.isdir(os.path.join(train_folder, train_sample_name))]  # noqa: E501

            val_folder = os.path.join(self._json_dir, 'val/')
            val_json_names = [val_sample_name + '.json'
                              for val_sample_name in os.listdir(val_folder)
                              if os.path.isdir(os.path.join(val_folder, val_sample_name))]  # noqa: E501

            test_folder = os.path.join(self._json_dir, 'test/')
            test_json_names = [test_sample_name + '.json'
                               for test_sample_name in os.listdir(test_folder)
                               if os.path.isdir(os.path.join(test_folder, test_sample_name))]  # noqa: E501

            return train_json_names, val_json_names, test_json_names

        train_idxs, val_idxs = train_test_split(range(len(json_names)),
                                                test_size=val_size)
        test_idxs = []
        if test_size is None:
            test_size = 0.0
        if test_size > 1e-8:
            train_idxs, test_idxs = train_test_split(
                train_idxs, test_size=test_size / (1 - val_size))
        train_json_names = [json_names[train_idx] for train_idx in train_idxs]
        val_json_names = [json_names[val_idx] for val_idx in val_idxs]
        test_json_names = [json_names[test_idx] for test_idx in test_idxs]

        return train_json_names, val_json_names, test_json_names

    def convert(self, val_size, test_size):
        json_names = [file_name for file_name in os.listdir(self._json_dir)
                      if os.path.isfile(os.path.join(self._json_dir, file_name)) and
                      file_name.endswith('.json')]
        folders = [file_name for file_name in os.listdir(self._json_dir)
                   if os.path.isdir(os.path.join(self._json_dir, file_name))]
        train_json_names, val_json_names, test_json_names = self._train_test_split(
            folders, json_names, val_size, test_size)

        self._make_train_val_dir()

        # convert labelme object to yolo format object, and save them to files
        # also get image from labelme json file and save them under images folder
        for target_dir, json_names in zip(('train/', 'val/', 'test/'),
                                          (train_json_names, val_json_names, test_json_names)):  # noqa: E501
            pool = Pool(NUM_THREADS)

            for json_name in json_names:
                pool.apply_async(self.covert_json_to_text,
                                 args=(target_dir, json_name))
            pool.close()
            pool.join()

        print('Generating dataset.yaml file ...')
        self._save_dataset_yaml()

    def covert_json_to_text(self, target_dir, json_name):
        json_path = os.path.join(self._json_dir, json_name)
        json_data = json.load(open(json_path))

        print('Converting %s for %s ...' %
              (json_name, target_dir.replace('/', '')))

        img_path = save_yolo_image(json_data,
                                   json_path,
                                   self._image_dir_path,
                                   target_dir)
        yolo_obj_list = self._get_yolo_object_list(json_data, img_path)
        save_yolo_label(json_name,
                        self._label_dir_path,
                        target_dir,
                        yolo_obj_list)

    def convert_one(self, json_name):
        json_path = os.path.join(self._json_dir, json_name)
        json_data = json.load(open(json_path))

        print('Converting %s ...' % json_name)

        img_path = save_yolo_image(json_data, json_name,
                                   self._json_dir, '')

        yolo_obj_list = self._get_yolo_object_list(json_data, img_path)
        save_yolo_label(json_name, self._json_dir,
                        '', yolo_obj_list)

    def _get_yolo_object_list(self, json_data, img_path):
        yolo_obj_list = []

        img_h, img_w, _ = cv2.imread(img_path).shape
        for shape in json_data["shapes"]:
            # labelme circle shape is different from others
            # it only has 2 points, 1st is circle center, 2nd is drag end point
            if shape['shape_type'] == 'circle':
                yolo_obj = self._get_circle_shape_yolo_object(
                    shape, img_h, img_w)
            else:
                yolo_obj = self._get_other_shape_yolo_object(
                    shape, img_h, img_w)

            yolo_obj_list.append(yolo_obj)

        return yolo_obj_list

    def _get_circle_shape_yolo_object(self, shape, img_h, img_w):
        obj_center_x, obj_center_y = shape['points'][0]

        radius = math.sqrt((obj_center_x - shape['points'][1][0]) ** 2 +
                           (obj_center_y - shape['points'][1][1]) ** 2)
        obj_w = 2 * radius
        obj_h = 2 * radius

        yolo_center_x = round(float(obj_center_x / img_w), 6)
        yolo_center_y = round(float(obj_center_y / img_h), 6)
        yolo_w = round(float(obj_w / img_w), 6)
        yolo_h = round(float(obj_h / img_h), 6)

        if shape['label'] in self._label_id_map:
            label_id = self._label_id_map[shape['label']]
        else:
            print('label %s not in %s' % shape['label'], self._label_list)

        return label_id, yolo_center_x, yolo_center_y, yolo_w, yolo_h

    def _get_other_shape_yolo_object(self, shape, img_h, img_w):

        point_list = shape['points']
        points = np.zeros(2 * len(point_list))
        points[::2] = [float(point[0]) / img_w for point in point_list]
        points[1::2] = [float(point[1]) / img_h for point in point_list]
        if len(points) == 4:
            if self._output_format == "polygon":
                points = extend_point_list(points)
            if self._output_format == "bbox":
                points = extend_point_list(points, "bbox")

        if shape['label'] in self._label_id_map:
            label_id = self._label_id_map[shape['label']]
        else:
            print('label %s not in %s' % shape['label'], self._label_list)

        return label_id, points.tolist()

    def _save_dataset_yaml(self):
        yaml_path = os.path.join(
            self._json_dir, 'YOLODataset/', 'dataset.yaml')

        with open(yaml_path, 'w+') as yaml_file:
            yaml_file.write('train: %s\n' %
                            os.path.join(self._image_dir_path, 'train/'))
            yaml_file.write('val: %s\n' %
                            os.path.join(self._image_dir_path, 'val/'))
            yaml_file.write('test: %s\n' %
                            os.path.join(self._image_dir_path, 'test/'))
            yaml_file.write('nc: %i\n' % len(self._label_id_map))

            names_str = ''

            for label, _ in self._label_id_map.items():
                names_str += "'%s', " % label
            names_str = names_str.rstrip(", ")
            yaml_file.write("names: [%s]" % names_str)
