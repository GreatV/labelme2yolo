use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::sync::OnceLock;

// Supported image formats
pub const IMG_FORMATS: &[&str] = &[
    "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm",
];

// Precomputed HashSet of image extensions for fast lookup
pub static IMAGE_EXTENSIONS_SET: OnceLock<HashSet<String>> = OnceLock::new();

/// Get the image extensions set
pub fn get_image_extensions_set() -> &'static HashSet<String> {
    IMAGE_EXTENSIONS_SET.get_or_init(|| IMG_FORMATS.iter().map(|ext| ext.to_lowercase()).collect())
}

// The Shape struct representing annotated shapes
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Shape {
    pub label: String,
    pub points: Vec<(f64, f64)>,
    pub group_id: Option<i64>,
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

// A wrapper for ImageAnnotation that supports streaming image_data
pub struct StreamingImageAnnotation {
    pub version: String,
    pub flags: Option<HashMap<String, bool>>,
    pub shapes: Vec<Shape>,
    pub image_path: String,
    pub image_data_stream: Option<Box<dyn Read + Send>>,
    pub image_height: u32,
    pub image_width: u32,
}

impl Clone for StreamingImageAnnotation {
    fn clone(&self) -> Self {
        // For the stream, we can't clone it directly, so we'll set it to None
        // This means cloning will lose the stream, but that's acceptable for our use case
        Self {
            version: self.version.clone(),
            flags: self.flags.clone(),
            shapes: self.shapes.clone(),
            image_path: self.image_path.clone(),
            image_data_stream: None,
            image_height: self.image_height,
            image_width: self.image_width,
        }
    }
}

impl StreamingImageAnnotation {
    /// Convert to regular ImageAnnotation by reading the image_data stream
    pub fn to_image_annotation(&mut self) -> std::io::Result<ImageAnnotation> {
        let image_data = if let Some(mut stream) = self.image_data_stream.take() {
            // Read the stream into a buffer
            let mut buffer = Vec::new();
            stream.read_to_end(&mut buffer)?;
            Some(
                String::from_utf8(buffer)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?,
            )
        } else {
            None
        };

        Ok(ImageAnnotation {
            version: self.version.clone(),
            flags: self.flags.clone(),
            shapes: self.shapes.clone(),
            image_path: self.image_path.clone(),
            image_data,
            image_height: self.image_height,
            image_width: self.image_width,
        })
    }
}

// Struct to hold processing statistics
#[derive(Debug, Default, Clone)]
pub struct ProcessingStats {
    pub total_files_processed: usize,
    pub successful_conversions: usize,
    pub skipped_missing_image: usize,
    pub skipped_no_image_data: usize,
    pub failed_conversions: usize,
}

impl ProcessingStats {
    pub fn new() -> Self {
        Self {
            total_files_processed: 0,
            successful_conversions: 0,
            skipped_missing_image: 0,
            skipped_no_image_data: 0,
            failed_conversions: 0,
        }
    }

    pub fn increment_total(&mut self) {
        self.total_files_processed += 1;
    }

    pub fn increment_successful(&mut self) {
        self.successful_conversions += 1;
    }

    pub fn increment_skipped_missing_image(&mut self) {
        self.skipped_missing_image += 1;
    }

    pub fn increment_skipped_no_image_data(&mut self) {
        self.skipped_no_image_data += 1;
    }

    pub fn increment_failed(&mut self) {
        self.failed_conversions += 1;
    }

    pub fn print_summary(&self) {
        log::info!("=== Processing Summary ===");
        log::info!("Total files processed: {}", self.total_files_processed);
        log::info!("Successful conversions: {}", self.successful_conversions);
        log::info!(
            "Skipped (missing image file): {}",
            self.skipped_missing_image
        );
        log::info!(
            "Skipped (no image data in JSON): {}",
            self.skipped_no_image_data
        );
        log::info!("Failed conversions: {}", self.failed_conversions);

        let total_skipped = self.skipped_missing_image + self.skipped_no_image_data;
        if total_skipped > 0 {
            log::warn!(
                "Total skipped annotations: {} (missing image file: {}, no image data: {})",
                total_skipped,
                self.skipped_missing_image,
                self.skipped_no_image_data
            );
        }
    }
}
