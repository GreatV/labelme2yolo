use indicatif::{ProgressBar, ProgressStyle};
use log::error;
use serde_json;
use std::fs;
use std::path::Path;

use crate::types::ImageAnnotation;

/// Helper function to infer image format from image bytes
pub fn infer_image_format(image_bytes: &[u8]) -> Option<&'static str> {
    if image_bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some("jpg")
    } else if image_bytes.starts_with(&[0x89, b'P', b'N', b'G']) {
        Some("png")
    } else if image_bytes.starts_with(b"BM") {
        Some("bmp")
    } else if image_bytes.starts_with(&[0x47, 0x49, 0x46]) {
        Some("gif")
    } else {
        None
    }
}

/// Read and parse a single JSON file into an ImageAnnotation struct using streaming
/// This prevents OOM errors by parsing JSON directly from a file stream instead of
/// loading the entire file into memory first.
pub fn read_and_parse_json(path: &Path) -> Option<ImageAnnotation> {
    // Open the file as a stream
    let file = match fs::File::open(path) {
        Ok(file) => file,
        Err(e) => {
            error!("Failed to open JSON file ({}): {:?}", path.display(), e);
            return None;
        }
    };

    // Parse JSON directly from the file stream
    match serde_json::from_reader(file) {
        Ok(annotation) => Some(annotation),
        Err(e) => {
            error!("Failed to parse JSON ({}): {:?}", path.display(), e);
            None
        }
    }
}

/// Create a progress bar with the given length and label
pub fn create_progress_bar(len: u64, label: &str) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(&format!(
                "{{spinner:.green}} [{}] [{{elapsed_precise}}] [{{bar:40.cyan/blue}}] {{pos}}/{{len}} ({{eta}})",
                label
            ))
            .progress_chars("#>-"),
    );
    pb
}

/// Safely create output directories and return their paths
pub fn create_output_directory(path: &Path) -> std::io::Result<std::path::PathBuf> {
    if path.exists() {
        log::warn!(
            "Directory {:?} already exists. Deleting and recreating it.",
            path
        );
        fs::remove_dir_all(path).and_then(|_| fs::create_dir_all(path))?;
    } else {
        fs::create_dir_all(path)?;
    }
    Ok(path.to_path_buf())
}
