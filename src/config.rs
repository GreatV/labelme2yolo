use clap::{Parser, ValueEnum};
use std::path::PathBuf;
use std::str::FromStr;

/// Command-line arguments parser for converting LabelMe JSON to YOLO format.
#[derive(Parser, Debug, Clone)]
#[command(version, long_about = None)]
pub struct Args {
    /// Directory containing LabelMe JSON files
    #[arg(short = 'd', long = "json_dir")]
    pub json_dir: String,

    /// Proportion of the dataset to use for validation
    #[arg(long = "val_size", default_value_t = 0.2, value_parser = validate_size)]
    pub val_size: f32,

    /// Proportion of the dataset to use for testing
    #[arg(long = "test_size", default_value_t = 0.0, value_parser = validate_size)]
    pub test_size: f32,

    /// Output format for YOLO annotations: 'bbox' or 'polygon'
    #[arg(
        long = "output_format",
        visible_alias = "format",
        value_enum,
        default_value = "bbox"
    )]
    pub output_format: Format,

    /// Seed for random shuffling
    #[arg(long = "seed", default_value_t = 42)]
    pub seed: u64,

    /// Flag to include images without annotations as background images
    #[arg(long = "include_background")]
    pub include_background: bool,

    /// List of labels in the dataset
    #[arg(use_value_delimiter = true)]
    pub label_list: Vec<String>,

    /// Number of worker threads for I/O operations (0 for automatic, defaults to min(4, physical cores))
    #[arg(long = "workers", default_value_t = 0)]
    pub workers: usize,

    /// Enable deterministic label mapping (slower but consistent IDs across runs)
    #[arg(long = "deterministic_labels")]
    pub deterministic_labels: bool,

    /// Buffer size for file I/O operations in KiB (64 or 128, default: 64)
    #[arg(long = "buffer_size_kib", default_value_t = 64, value_parser = validate_buffer_size)]
    pub buffer_size_kib: usize,

    // COCO-specific arguments (only used by labelme2coco)
    /// Segmentation mode for COCO export: polygon (default) or bbox-only
    #[arg(long = "segmentation", default_value = "polygon")]
    pub segmentation_mode: SegmentationMode,

    /// Starting IDs for COCO export (format: image=N,ann=M, default: image=1,ann=1)
    #[arg(long = "start_ids", default_value = "image=1,ann=1")]
    pub start_ids: String,

    /// Source for categories: label_list, inferred (default), or file=path/to/categories.txt
    #[arg(long = "categories_from", default_value = "inferred")]
    pub categories_from: String,
}

// Enumeration for the YOLO output format
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
pub enum Format {
    Polygon,
    Bbox,
}

// Enumeration for the COCO segmentation mode
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
pub enum SegmentationMode {
    Polygon,
    BboxOnly,
}

// Validate that the size is between 0.0 and 1.0
fn validate_size(s: &str) -> Result<f32, String> {
    match f32::from_str(s) {
        Ok(val) if (0.0..=1.0).contains(&val) => Ok(val),
        _ => Err("SIZE must be between 0.0 and 1.0".to_string()),
    }
}

// Validate that the buffer size is 64 or 128 KiB
fn validate_buffer_size(s: &str) -> Result<usize, String> {
    match usize::from_str(s) {
        Ok(val) if val == 64 || val == 128 => Ok(val),
        Ok(_) => Err("BUFFER_SIZE must be either 64 or 128".to_string()),
        Err(_) => Err("Invalid buffer size value".to_string()),
    }
}

/// COCO-specific configuration extracted from Args
#[derive(Debug, Clone)]
pub struct CocoConfig {
    pub segmentation_mode: SegmentationMode,
    pub start_image_id: u32,
    pub start_annotation_id: u32,
    pub categories_source: CategoriesSource,
}

/// Source for categories in COCO export
#[derive(Debug, Clone)]
pub enum CategoriesSource {
    LabelList,
    Inferred,
    File(PathBuf),
}

/// Parse start_ids string in format "image=N,ann=M"
pub fn parse_start_ids(start_ids: &str) -> Result<(u32, u32), String> {
    let parts: Vec<&str> = start_ids.split(',').collect();
    if parts.len() != 2 {
        return Err("start_ids must be in format 'image=N,ann=M'".to_string());
    }

    let image_id = parts[0]
        .strip_prefix("image=")
        .ok_or("Expected 'image=N' in first part")?
        .parse::<u32>()
        .map_err(|_| "Invalid image ID")?;

    let ann_id = parts[1]
        .strip_prefix("ann=")
        .ok_or("Expected 'ann=M' in second part")?
        .parse::<u32>()
        .map_err(|_| "Invalid annotation ID")?;

    Ok((image_id, ann_id))
}

/// Parse categories_from string
pub fn parse_categories_from(categories_from: &str) -> Result<CategoriesSource, String> {
    if categories_from == "label_list" {
        Ok(CategoriesSource::LabelList)
    } else if categories_from == "inferred" {
        Ok(CategoriesSource::Inferred)
    } else if let Some(file_path) = categories_from.strip_prefix("file=") {
        Ok(CategoriesSource::File(PathBuf::from(file_path)))
    } else {
        Err("categories_from must be 'label_list', 'inferred', or 'file=path'".to_string())
    }
}

impl Args {
    /// Extract COCO-specific configuration from Args
    pub fn to_coco_config(&self) -> Result<CocoConfig, String> {
        let (start_image_id, start_annotation_id) = parse_start_ids(&self.start_ids)?;
        let categories_source = parse_categories_from(&self.categories_from)?;

        Ok(CocoConfig {
            segmentation_mode: self.segmentation_mode,
            start_image_id,
            start_annotation_id,
            categories_source,
        })
    }
}
