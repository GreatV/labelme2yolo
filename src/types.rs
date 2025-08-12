use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Supported image formats
pub const IMG_FORMATS: &[&str] = &[
    "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm",
];

// The Shape struct representing annotated shapes
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Shape {
    pub label: String,
    pub points: Vec<(f64, f64)>,
    pub group_id: Option<String>,
    pub shape_type: String,
    pub description: Option<String>,
    pub mask: Option<String>,
}

// The ImageAnnotation struct representing the annotation information of an image
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ImageAnnotation {
    pub version: String,
    pub flags: Option<HashMap<String, bool>>,
    pub shapes: Vec<Shape>,
    pub image_path: String,
    pub image_data: Option<String>,
    pub image_height: u32,
    pub image_width: u32,
}

// Struct to hold the paths to the output directories for train/val/test splits
pub struct OutputDirs {
    pub train_labels_dir: std::path::PathBuf,
    pub val_labels_dir: std::path::PathBuf,
    pub train_images_dir: std::path::PathBuf,
    pub val_images_dir: std::path::PathBuf,
    pub test_labels_dir: Option<std::path::PathBuf>,
    pub test_images_dir: Option<std::path::PathBuf>,
}

// Struct to hold the split datasets for training, validation, and testing
pub struct SplitData {
    pub train_annotations: Vec<(std::path::PathBuf, Option<ImageAnnotation>)>,
    pub val_annotations: Vec<(std::path::PathBuf, Option<ImageAnnotation>)>,
    pub test_annotations: Vec<(std::path::PathBuf, Option<ImageAnnotation>)>,
}
