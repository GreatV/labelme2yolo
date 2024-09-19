use std::collections::{HashMap, HashSet};
use std::fs::{self, copy, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::{
    atomic::{AtomicUsize, Ordering::Relaxed},
    Arc, Mutex,
};

use clap::{Parser, ValueEnum};
use env_logger;
use glob::glob;
use indicatif::{ProgressBar, ProgressStyle};
use log::{error, info, warn};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json;

// Supported image formats
const IMG_FORMATS: &[&str] = &[
    "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm",
];

/// Helper function to infer image format from image bytes
fn infer_image_format(image_bytes: &[u8]) -> Option<&'static str> {
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

// The Shape struct representing annotated shapes
#[derive(Debug, Serialize, Deserialize, Clone)]
struct Shape {
    label: String,
    points: Vec<(f64, f64)>,
    group_id: Option<String>,
    shape_type: String,
    description: Option<String>,
    mask: Option<String>,
}

// The ImageAnnotation struct representing the annotation information of an image
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct ImageAnnotation {
    version: String,
    flags: Option<HashMap<String, bool>>,
    shapes: Vec<Shape>,
    image_path: String,
    image_data: Option<String>,
    image_height: u32,
    image_width: u32,
}

/// Command-line arguments parser for converting LabelMe JSON to YOLO format.
#[derive(Parser, Debug)]
#[command(version, long_about = None)]
struct Args {
    /// Directory containing LabelMe JSON files
    #[arg(short = 'd', long = "json_dir")]
    json_dir: String,

    /// Proportion of the dataset to use for validation
    #[arg(long = "val_size", default_value_t = 0.2, value_parser = validate_size)]
    val_size: f32,

    /// Proportion of the dataset to use for testing
    #[arg(long = "test_size", default_value_t = 0.0, value_parser = validate_size)]
    test_size: f32,

    /// Output format for YOLO annotations: 'bbox' or 'polygon'
    #[arg(
        long = "output_format",
        visible_alias = "format",
        value_enum,
        default_value = "bbox"
    )]
    output_format: Format,

    /// Seed for random shuffling
    #[arg(long = "seed", default_value_t = 42)]
    seed: u64,

    /// Flag to include images without annotations as background images
    #[arg(long = "include_background")]
    include_background: bool,

    /// List of labels in the dataset
    #[arg(use_value_delimiter = true)]
    label_list: Vec<String>,
}

// Enumeration for the YOLO output format
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum Format {
    Polygon,
    Bbox,
}

// Validate that the size is between 0.0 and 1.0
fn validate_size(s: &str) -> Result<f32, String> {
    match f32::from_str(s) {
        Ok(val) if val >= 0.0 && val <= 1.0 => Ok(val),
        _ => Err("SIZE must be between 0.0 and 1.0".to_string()),
    }
}

fn main() {
    // Initialize the logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    let dirname = PathBuf::from(&args.json_dir);
    if !dirname.exists() {
        error!("The specified json_dir does not exist: {}", args.json_dir);
        return;
    }

    info!("Starting the conversion process...");

    match setup_output_directories(&args, &dirname) {
        Ok(output_dirs) => {
            let mut annotations = read_and_parse_json_files(&dirname, &args);
            info!("Read and parsed {} files.", annotations.len());

            let split_data =
                split_annotations(&mut annotations, args.val_size, args.test_size, args.seed);

            let label_map = Arc::new(Mutex::new(HashMap::new()));
            let next_class_id = Arc::new(AtomicUsize::new(0));

            // Preinitialize the label map with all possible labels from the dataset
            initialize_label_map(&split_data, &label_map, &next_class_id, &args);

            process_all_annotations(&split_data, &output_dirs, &label_map, &args, &dirname);

            info!("Creating dataset.yaml file...");
            if let Err(e) = create_dataset_yaml(&dirname, &args, &label_map) {
                error!("Failed to create dataset.yaml: {}", e);
            } else {
                info!("Conversion process completed successfully.");
            }
        }
        Err(e) => error!("Failed to set up output directories: {}", e),
    }
}

// Struct to hold the paths to the output directories for train/val/test splits
struct OutputDirs {
    train_labels_dir: PathBuf,
    val_labels_dir: PathBuf,
    train_images_dir: PathBuf,
    val_images_dir: PathBuf,
    test_labels_dir: Option<PathBuf>,
    test_images_dir: Option<PathBuf>,
}

// Safely create output directories and return their paths
fn create_output_directory(path: &Path) -> std::io::Result<PathBuf> {
    if path.exists() {
        warn!(
            "Directory {:?} already exists. Deleting and recreating it.",
            path
        );
        fs::remove_dir_all(path).and_then(|_| fs::create_dir_all(path))?;
    } else {
        fs::create_dir_all(path)?;
    }
    Ok(path.to_path_buf())
}

// Set up the directory structure for YOLO dataset output
fn setup_output_directories(args: &Args, dirname: &Path) -> std::io::Result<OutputDirs> {
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

// Read and parse JSON files, and handle images without annotations efficiently
fn read_and_parse_json_files(
    dirname: &Path,
    args: &Args,
) -> Vec<(PathBuf, Option<ImageAnnotation>)> {
    // Collect all JSON files
    let json_pattern = format!("{}/**/*.json", dirname.display());
    let json_entries: Vec<_> = glob(&json_pattern)
        .expect("Failed to read JSON glob pattern")
        .filter_map(|entry| entry.ok())
        .collect();

    // Vector to store image paths and their annotations
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
        // Collect all image files
        let mut image_entries = Vec::new();
        for ext in IMG_FORMATS {
            let pattern = format!("{}/**/*.{}", dirname.display(), ext);
            image_entries.extend(
                glob(&pattern)
                    .expect("Failed to read image glob pattern")
                    .filter_map(|entry| entry.ok()),
            );
        }

        // Create a set of image paths that have annotations
        let annotated_images: HashSet<_> =
            annotations.iter().map(|(path, _)| path.clone()).collect();

        // Add background images
        let background_images: Vec<_> = image_entries
            .into_par_iter()
            .filter(|image_path| !annotated_images.contains(image_path))
            .map(|image_path| (image_path, None))
            .collect();

        annotations.extend(background_images);
    }

    annotations
}

// Read and parse a single JSON file into an ImageAnnotation struct
fn read_and_parse_json(path: &Path) -> Option<ImageAnnotation> {
    fs::read_to_string(path).ok().and_then(|content| {
        serde_json::from_str::<ImageAnnotation>(&content)
            .map_err(|e| error!("Failed to parse JSON ({}): {:?}", path.display(), e))
            .ok()
    })
}

// Struct to hold the split datasets for training, validation, and testing
struct SplitData {
    train_annotations: Vec<(PathBuf, Option<ImageAnnotation>)>,
    val_annotations: Vec<(PathBuf, Option<ImageAnnotation>)>,
    test_annotations: Vec<(PathBuf, Option<ImageAnnotation>)>,
}

// Split the annotations into training, validation, and testing sets
fn split_annotations(
    annotations: &mut Vec<(PathBuf, Option<ImageAnnotation>)>,
    val_size: f32,
    test_size: f32,
    seed: u64,
) -> SplitData {
    let mut rng = StdRng::seed_from_u64(seed);
    annotations.shuffle(&mut rng);

    let test_size = (annotations.len() as f32 * test_size).ceil() as usize;
    let val_size = (annotations.len() as f32 * val_size).ceil() as usize;

    let test_annotations = annotations.drain(0..test_size).collect();
    let val_annotations = annotations.drain(0..val_size).collect();
    let train_annotations = annotations.to_vec();

    SplitData {
        train_annotations,
        val_annotations,
        test_annotations,
    }
}

// Initialize the label map with labels found in the dataset or specified in label_list
fn initialize_label_map(
    split_data: &SplitData,
    label_map: &Arc<Mutex<HashMap<String, usize>>>,
    next_class_id: &Arc<AtomicUsize>,
    args: &Args,
) {
    let mut map = label_map.lock().unwrap();

    // If label_list is specified, use it to initialize label_map
    if !args.label_list.is_empty() {
        for (id, label) in args.label_list.iter().enumerate() {
            map.insert(label.clone(), id);
        }
        next_class_id.store(args.label_list.len(), Relaxed);
    } else {
        // Otherwise, use labels found in the dataset
        split_data
            .train_annotations
            .iter()
            .chain(split_data.val_annotations.iter())
            .chain(split_data.test_annotations.iter())
            .filter_map(|(_, annotation)| annotation.as_ref())
            .flat_map(|annotation| annotation.shapes.iter())
            .for_each(|shape| {
                if !map.contains_key(&shape.label) {
                    let new_id = next_class_id.fetch_add(1, Relaxed);
                    map.insert(shape.label.clone(), new_id);
                }
            });
    }
}

fn process_all_annotations(
    split_data: &SplitData,
    output_dirs: &OutputDirs,
    label_map: &Arc<Mutex<HashMap<String, usize>>>,
    args: &Args,
    dirname: &Path,
) {
    let train_pb = create_progress_bar(split_data.train_annotations.len() as u64, "Train");
    process_annotations_in_parallel(
        &split_data.train_annotations,
        &output_dirs.train_labels_dir,
        &output_dirs.train_images_dir,
        label_map,
        args,
        dirname,
        &train_pb,
    );
    train_pb.finish_with_message("Train processing complete");

    let val_pb = create_progress_bar(split_data.val_annotations.len() as u64, "Val");
    process_annotations_in_parallel(
        &split_data.val_annotations,
        &output_dirs.val_labels_dir,
        &output_dirs.val_images_dir,
        label_map,
        args,
        dirname,
        &val_pb,
    );
    val_pb.finish_with_message("Val processing complete");

    if let (Some(test_labels_dir), Some(test_images_dir)) =
        (&output_dirs.test_labels_dir, &output_dirs.test_images_dir)
    {
        let test_pb = create_progress_bar(split_data.test_annotations.len() as u64, "Test");
        process_annotations_in_parallel(
            &split_data.test_annotations,
            test_labels_dir,
            test_images_dir,
            label_map,
            args,
            dirname,
            &test_pb,
        );
        test_pb.finish_with_message("Test processing complete");
    }
}

// Create a progress bar with the given length and label
fn create_progress_bar(len: u64, label: &str) -> ProgressBar {
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

// Process a batch of annotations in parallel
fn process_annotations_in_parallel(
    annotations: &[(PathBuf, Option<ImageAnnotation>)],
    labels_dir: &Path,
    images_dir: &Path,
    label_map: &Arc<Mutex<HashMap<String, usize>>>,
    args: &Args,
    base_dir: &Path,
    pb: &ProgressBar,
) {
    annotations.par_iter().for_each(|(image_path, annotation)| {
        if let Err(e) = process_annotation(
            image_path,
            annotation.as_ref(),
            labels_dir,
            images_dir,
            label_map,
            args,
            base_dir,
        ) {
            error!(
                "Failed to process annotation {}: {}",
                image_path.display(),
                e
            );
        }
        pb.inc(1);
    });
}

// Process a single annotation and convert it to YOLO format
fn process_annotation(
    image_path: &Path,
    annotation: Option<&ImageAnnotation>,
    labels_dir: &Path,
    images_dir: &Path,
    label_map: &Arc<Mutex<HashMap<String, usize>>>,
    args: &Args,
    _base_dir: &Path,
) -> std::io::Result<()> {
    let sanitized_name =
        sanitize_filename::sanitize(image_path.file_stem().unwrap().to_str().unwrap());

    if image_path.exists() {
        // Copy the image file
        let image_extension = image_path.extension().unwrap_or_default();
        let image_output_path = images_dir
            .join(&sanitized_name)
            .with_extension(image_extension);
        copy(&image_path, &image_output_path)?;
    } else if let Some(annotation) = annotation {
        // Handle missing image file by extracting image_data from JSON
        if let Some(image_data) = &annotation.image_data {
            if !image_data.is_empty() {
                // Decode and save the image data from JSON
                let image_bytes = base64::decode(image_data)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                let image_extension = infer_image_format(&image_bytes).unwrap_or("png");
                let image_output_path = images_dir
                    .join(&sanitized_name)
                    .with_extension(image_extension);
                let mut file = File::create(&image_output_path)?;
                file.write_all(&image_bytes)?;
            } else {
                // No image data available
                warn!("No image data found in JSON for: {:?}", image_path);
                return Ok(());
            }
        } else {
            // No image data available
            warn!("No image data found in JSON for: {:?}", image_path);
            return Ok(());
        }
    } else {
        // No image file or annotation available
        warn!(
            "Image file not found and no annotation provided for: {:?}",
            image_path
        );
        return Ok(());
    }

    // Generate label file
    let label_output_path = labels_dir.join(&sanitized_name).with_extension("txt");
    let mut writer = BufWriter::new(File::create(&label_output_path)?);

    if let Some(annotation) = annotation {
        // Convert annotation to YOLO format
        let yolo_data = convert_to_yolo_format(annotation, args, label_map);
        writer.write_all(yolo_data.as_bytes())?;
    } else if args.include_background {
        // Background image, generate empty label file
        writer.write_all(b"")?;
    }

    Ok(())
}

// Convert an annotation to YOLO format (bounding box or polygon)
fn convert_to_yolo_format(
    annotation: &ImageAnnotation,
    args: &Args,
    label_map: &Arc<Mutex<HashMap<String, usize>>>,
) -> String {
    let mut yolo_data = String::with_capacity(annotation.shapes.len() * 64);

    for shape in &annotation.shapes {
        let class_id = match label_map.lock().unwrap().get(&shape.label) {
            Some(class_id) => *class_id,
            None => continue,
        };

        match args.output_format {
            Format::Polygon => {
                yolo_data.push_str(&format!("{}", class_id));
                process_polygon_shape(&mut yolo_data, annotation, shape);
                yolo_data.push('\n');
            }
            Format::Bbox => {
                let (x_center, y_center, width, height) = calculate_bounding_box(annotation, shape);
                yolo_data.push_str(&format!(
                    "{} {:.6} {:.6} {:.6} {:.6}\n",
                    class_id, x_center, y_center, width, height
                ));
            }
        }
    }

    yolo_data
}

// Process polygon shape data for YOLO format
fn process_polygon_shape(yolo_data: &mut String, annotation: &ImageAnnotation, shape: &Shape) {
    if shape.shape_type == "rectangle" {
        let (x1, y1) = shape.points[0];
        let (x2, y2) = shape.points[1];
        let rect_points = vec![(x1, y1), (x2, y1), (x2, y2), (x1, y2)];
        for &(x, y) in &rect_points {
            let x_norm = x / annotation.image_width as f64;
            let y_norm = y / annotation.image_height as f64;
            yolo_data.push_str(&format!(" {:.6} {:.6}", x_norm, y_norm));
        }
    } else if shape.shape_type == "circle" {
        const CIRCLE_POINTS: usize = 12;
        let (cx, cy) = shape.points[0];
        let (px, py) = shape.points[1];
        let radius = ((cx - px).powi(2) + (cy - py).powi(2)).sqrt();
        for i in 0..CIRCLE_POINTS {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / CIRCLE_POINTS as f64;
            let x = cx + radius * angle.cos();
            let y = cy + radius * angle.sin();
            let x_norm = x / annotation.image_width as f64;
            let y_norm = y / annotation.image_height as f64;
            yolo_data.push_str(&format!(" {:.6} {:.6}", x_norm, y_norm));
        }
    } else {
        for &(x, y) in &shape.points {
            let x_norm = x / annotation.image_width as f64;
            let y_norm = y / annotation.image_height as f64;
            yolo_data.push_str(&format!(" {:.6} {:.6}", x_norm, y_norm));
        }
    }
}

// Calculate bounding box for YOLO format
fn calculate_bounding_box(annotation: &ImageAnnotation, shape: &Shape) -> (f64, f64, f64, f64) {
    let (x_min, y_min, x_max, y_max) = if shape.shape_type == "circle" {
        let (cx, cy) = shape.points[0];
        let (px, py) = shape.points[1];
        let radius = ((cx - px).powi(2) + (cy - py).powi(2)).sqrt();
        (cx - radius, cy - radius, cx + radius, cy + radius)
    } else {
        shape.points.iter().fold(
            (f64::MAX, f64::MAX, f64::MIN, f64::MIN),
            |(x_min, y_min, x_max, y_max), &(x, y)| {
                (x_min.min(x), y_min.min(y), x_max.max(x), y_max.max(y))
            },
        )
    };

    let x_center = (x_min + x_max) / 2.0 / annotation.image_width as f64;
    let y_center = (y_min + y_max) / 2.0 / annotation.image_height as f64;
    let width = (x_max - x_min) / annotation.image_width as f64;
    let height = (y_max - y_min) / annotation.image_height as f64;

    (x_center, y_center, width, height)
}

// Create the dataset.yaml file for YOLO training
fn create_dataset_yaml(
    dirname: &Path,
    args: &Args,
    label_map: &Arc<Mutex<HashMap<String, usize>>>,
) -> std::io::Result<()> {
    let dataset_yaml_path = dirname.join("YOLODataset/dataset.yaml");
    let mut dataset_yaml = BufWriter::new(File::create(&dataset_yaml_path)?);
    let absolute_path = fs::canonicalize(&dirname.join("YOLODataset"))?;
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
        .lock()
        .unwrap()
        .iter()
        .map(|(label, id)| (label.clone(), *id))
        .collect();
    sorted_labels.sort_by_key(|&(_, id)| id);

    for (label, id) in sorted_labels {
        yaml_content.push_str(&format!("    {}: {}\n", id, label));
    }
    dataset_yaml.write_all(yaml_content.as_bytes())
}
