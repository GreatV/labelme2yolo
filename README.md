**Forked from [rooneysh/Labelme2YOLO](https://github.com/rooneysh/Labelme2YOLO)**

# Labelme2YOLO

[![PyPI - Version](https://img.shields.io/pypi/v/labelme2yolo.svg)](https://pypi.org/project/labelme2yolo)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/labelme2yolo.svg)](https://pypi.org/project/labelme2yolo)

Help converting LabelMe Annotation Tool JSON format to YOLO text file format. 
If you've already marked your segmentation dataset by LabelMe, it's easy to use this tool to help converting to YOLO format dataset.

## Installation

```console
pip install labelme2yolo
```

## Parameters Explain
**--json_dir** LabelMe JSON files folder path.

**--val_size (Optional)** Validation dataset size, for example 0.2 means 20% for validation.

**--test_size (Optional)** Test dataset size, for example 0.2 means 20% for Test.

**--json_name (Optional)** Convert single LabelMe JSON file.

## How to Use

### 1. Convert JSON files, split training, validation and test dataset by --val_size and --test_size
Put all LabelMe JSON files under **labelme_json_dir**, and run this python command.
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
If you already split train dataset and validation dataset for LabelMe by yourself, please put these folder under labelme_json_dir, for example,
```bash
/path/to/labelme_json_dir/train/
/path/to/labelme_json_dir/val/
```
Put all LabelMe JSON files under **labelme_json_dir**. 
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
Put LabelMe JSON file under **labelme_json_dir**. , and run this python command.
```bash
labelme2yolo --json_dir /path/to/labelme_json_dir/ --json_name 2.json
```
Script would generate YOLO format text label and image under **labelme_json_dir**, for example,
```bash
/path/to/labelme_json_dir/2.text
/path/to/labelme_json_dir/2.png
```


## License

`labelme2yolo` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
