# Labelme2YOLO

**Forked from [rooneysh/Labelme2YOLO](https://github.com/rooneysh/Labelme2YOLO)**

[![PyPI - Version](https://img.shields.io/pypi/v/labelme2yolo.svg)](https://pypi.org/project/labelme2yolo)
![PyPI - Downloads](https://img.shields.io/pypi/dm/labelme2yolo?style=flat)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/labelme2yolo.svg)](https://pypi.org/project/labelme2yolo)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/12122fe86f8643c4aa5667c20d528f61)](https://www.codacy.com/gh/GreatV/labelme2yolo/dashboard?utm_source=github.com\&utm_medium=referral\&utm_content=GreatV/labelme2yolo\&utm_campaign=Badge_Grade)

Help converting LabelMe Annotation Tool JSON format to YOLO text file format.
If you've already marked your segmentation dataset by LabelMe, it's easy to use this tool to help converting to YOLO format dataset.

## New

* export data as yolo polygon annotation (for YOLOv5 v7.0 segmentation)
* Now you can choose the output format of the label text. The two available alternatives are `polygon` and bounding box (`bbox`).

## Installation

```console
pip install labelme2yolo
```

## Parameters Explain

**--json\_dir** LabelMe JSON files folder path.

**--val\_size (Optional)** Validation dataset size, for example 0.2 means 20% for validation.

**--test\_size (Optional)** Test dataset size, for example 0.2 means 20% for Test.

**--json\_name (Optional)** Convert single LabelMe JSON file.

**--output\_format (Optional)** The output format of label.

**--label\_list (Optional)** The pre-assigned category labels.

## How to Use

### 1. Convert JSON files, split training, validation and test dataset by --val\_size and --test\_size

Put all LabelMe JSON files under **labelme\_json\_dir**, and run this python command.

```bash
labelme2yolo --json_dir /path/to/labelme_json_dir/ --val_size 0.15 --test_size 0.15
```

Script would generate YOLO format dataset labels and images under different folders, for example,

```bash
/path/to/labelme_json_dir/YOLODataset/labels/train/
/path/to/labelme_json_dir/YOLODataset/labels/test/
/path/to/labelme_json_dir/YOLODataset/labels/val/
/path/to/labelme_json_dir/YOLODataset/images/train/
/path/to/labelme_json_dir/YOLODataset/images/test/
/path/to/labelme_json_dir/YOLODataset/images/val/

/path/to/labelme_json_dir/YOLODataset/dataset.yaml
```

### 2. Convert JSON files, split training and validation dataset by folder

If you already split train dataset and validation dataset for LabelMe by yourself, please put these folder under labelme\_json\_dir, for example,

```bash
/path/to/labelme_json_dir/train/
/path/to/labelme_json_dir/val/
```

Put all LabelMe JSON files under **labelme\_json\_dir**.
Script would read train and validation dataset by folder.
Run this python command.

```bash
labelme2yolo --json_dir /path/to/labelme_json_dir/
```

Script would generate YOLO format dataset labels and images under different folders, for example,

```bash
/path/to/labelme_json_dir/YOLODataset/labels/train/
/path/to/labelme_json_dir/YOLODataset/labels/val/
/path/to/labelme_json_dir/YOLODataset/images/train/
/path/to/labelme_json_dir/YOLODataset/images/val/

/path/to/labelme_json_dir/YOLODataset/dataset.yaml
```

### 3. Convert single JSON file

Put LabelMe JSON file under **labelme\_json\_dir**. , and run this python command.

```bash
labelme2yolo --json_dir /path/to/labelme_json_dir/ --json_name 2.json
```

Script would generate YOLO format text label and image under **labelme\_json\_dir**, for example,

```bash
/path/to/labelme_json_dir/2.text
/path/to/labelme_json_dir/2.png
```

## How to build package/wheel

1. [install hatch](https://hatch.pypa.io/latest/install/)
2. Run the following command:

```shell
hatch build
```

## License

`labelme2yolo` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
