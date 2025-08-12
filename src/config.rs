use clap::{Parser, ValueEnum};
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
}

// Enumeration for the YOLO output format
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
pub enum Format {
    Polygon,
    Bbox,
}

// Validate that the size is between 0.0 and 1.0
fn validate_size(s: &str) -> Result<f32, String> {
    match f32::from_str(s) {
        Ok(val) if (0.0..=1.0).contains(&val) => Ok(val),
        _ => Err("SIZE must be between 0.0 and 1.0".to_string()),
    }
}
