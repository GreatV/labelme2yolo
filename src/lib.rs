//! LabelMe to YOLO format converter
//!
//! This library provides functionality to convert LabelMe JSON annotations to YOLO format
//! for object detection training.

pub mod coco;
pub mod coco_dataset;
pub mod config;
pub mod conversion;
pub mod io;
pub mod streaming_json;
pub mod types;
pub mod utils;
pub mod yolo_dataset;

// Re-export commonly used types and functions
pub use config::{Args, Format};
pub use io::{process_background_images, process_json_files_streaming, setup_output_directories};
pub use types::{ImageAnnotation, OutputDirs, Shape, SplitData};
pub use yolo_dataset::process_dataset;

// COCO-specific exports
pub use coco::{CocoConfig, CocoWriter};
pub use coco_dataset::{process_coco_dataset, setup_coco_output_directories};
