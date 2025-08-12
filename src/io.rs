use glob::glob;
use rayon::prelude::*;
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::config::Args;
use crate::types::{ImageAnnotation, OutputDirs, IMG_FORMATS};
use crate::utils::{create_output_directory, read_and_parse_json};

/// Set up the directory structure for YOLO dataset output
pub fn setup_output_directories(args: &Args, dirname: &Path) -> std::io::Result<OutputDirs> {
    let labels_dir = create_output_directory(&dirname.join("YOLODataset/labels"))?;
    let images_dir = create_output_directory(&dirname.join("YOLODataset/images"))?;

    let train_labels_dir = create_output_directory(&labels_dir.join("train"))?;
    let val_labels_dir = create_output_directory(&labels_dir.join("val"))?;
    let train_images_dir = create_output_directory(&images_dir.join("train"))?;
    let val_images_dir = create_output_directory(&images_dir.join("val"))?;

    let (test_labels_dir, test_images_dir) = if args.test_size > 0.0 {
        (
            Some(create_output_directory(&labels_dir.join("test"))?),
            Some(create_output_directory(&images_dir.join("test"))?),
        )
    } else {
        (None, None)
    };

    Ok(OutputDirs {
        train_labels_dir,
        val_labels_dir,
        train_images_dir,
        val_images_dir,
        test_labels_dir,
        test_images_dir,
    })
}

/// Read and parse JSON files, and handle images without annotations efficiently
pub fn read_and_parse_json_files(
    dirname: &Path,
    args: &Args,
) -> Vec<(PathBuf, Option<ImageAnnotation>)> {
    // Collect all JSON files in parallel
    let json_pattern = format!("{}/**/*.json", dirname.display());
    let json_entries: Vec<_> = glob(&json_pattern)
        .expect("Failed to read JSON glob pattern")
        .filter_map(|entry| entry.ok())
        .collect();

    // Vector to store image paths and their annotations - process in parallel
    let annotations_from_json: Vec<_> = json_entries
        .into_par_iter()
        .filter_map(|json_path| {
            if let Some(annotation) = read_and_parse_json(&json_path) {
                // Determine the image path
                let image_path = dirname.join(&annotation.image_path);
                Some((image_path, Some(annotation)))
            } else {
                None
            }
        })
        .collect();

    let mut annotations = annotations_from_json;

    // Include background images if the flag is set
    if args.include_background {
        // Collect all image files in parallel
        let image_entries: Vec<_> = IMG_FORMATS
            .par_iter()
            .flat_map(|ext| {
                let pattern = format!("{}/**/*.{}", dirname.display(), ext);
                glob(&pattern)
                    .expect("Failed to read image glob pattern")
                    .filter_map(|entry| entry.ok())
                    .collect::<Vec<_>>()
            })
            .collect();

        // Create a set of image paths that have annotations
        let annotated_images: HashSet<_> =
            annotations.iter().map(|(path, _)| path.clone()).collect();

        // Add background images in parallel
        let background_images: Vec<_> = image_entries
            .into_par_iter()
            .filter(|image_path| !annotated_images.contains(image_path))
            .map(|image_path| (image_path, None))
            .collect();

        annotations.extend(background_images);
    }

    annotations
}

/// Create the dataset.yaml file for YOLO training
pub fn create_dataset_yaml(
    dirname: &Path,
    args: &Args,
    label_map: &dashmap::DashMap<String, usize>,
) -> std::io::Result<()> {
    let dataset_yaml_path = dirname.join("YOLODataset/dataset.yaml");
    let mut dataset_yaml = BufWriter::new(File::create(&dataset_yaml_path)?);
    let absolute_path = fs::canonicalize(dirname.join("YOLODataset"))?;
    let mut yaml_content = format!(
        "path: {}\ntrain: images/train\nval: images/val\n",
        absolute_path.to_string_lossy()
    );
    if args.test_size > 0.0 {
        yaml_content.push_str("test: images/test\n");
    } else {
        yaml_content.push_str("test:\n");
    }
    yaml_content.push_str("\nnames:\n");

    // Extract and sort labels by their ID
    let mut sorted_labels: Vec<_> = label_map
        .iter()
        .map(|entry| (entry.key().clone(), *entry.value()))
        .collect();
    sorted_labels.sort_by_key(|&(_, id)| id);

    for (label, id) in sorted_labels {
        yaml_content.push_str(&format!("    {}: {}\n", id, label));
    }
    dataset_yaml.write_all(yaml_content.as_bytes())
}
