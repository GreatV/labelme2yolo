//! LabelMe to YOLO format converter
//!
//! This library provides functionality to convert LabelMe JSON annotations to YOLO format
//! for object detection training.

pub mod config;
pub mod conversion;
pub mod dataset;
pub mod io;
pub mod types;
pub mod utils;

// Re-export commonly used types and functions
pub use config::{Args, Format};
pub use dataset::process_dataset;
pub use io::{read_and_parse_json_files, setup_output_directories};
pub use types::{ImageAnnotation, OutputDirs, Shape, SplitData};
