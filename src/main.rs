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
use std::collections::HashMap;
use std::fs;
use std::fs::copy;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::{Arc, Mutex};

#[derive(Debug, Serialize, Deserialize)]
struct Shape {
    label: String,
    points: Vec<(f64, f64)>,
    group_id: Option<String>,
    shape_type: String,
    description: Option<String>,
    mask: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
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

/// A powerful tool for converting LabelMe's JSON format to YOLO dataset format.  
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The dir of the labelme json files  
    #[arg(short = 'd', long = "json_dir")]
    json_dir: String,

    /// The validation dataset size  
    #[arg(long = "val_size", default_value_t = 0.2, value_parser = validate_size)]
    val_size: f32,

    /// The test dataset size  
    #[arg(long = "test_size", default_value_t = 0.0, value_parser = validate_size)]
    test_size: f32,

    /// The output format of yolo  
    #[arg(
        long = "output_format",
        visible_alias = "format",
        value_enum,
        default_value = "bbox"
    )]
    output_format: Format,

    /// The ordered label list  
    #[arg(use_value_delimiter = true)]
    label_list: Vec<String>,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum Format {
    /// Output as polygon format  
    Polygon,
    /// Output as bounding-box format  
    Bbox,
}

fn validate_size(s: &str) -> Result<f32, String> {
    match f32::from_str(s) {
        Ok(val) if val >= 0.0 && val <= 1.0 => Ok(val),
        _ => Err("SIZE must be between 0.0 and 1.0".to_string()),
    }
}

fn create_dir(path: &Path) {
    if path.exists() {
        fs::remove_dir_all(path).expect("Failed to remove existing directory");
    }
    fs::create_dir_all(path).expect("Failed to create directory");
}

fn read_and_parse_json(path: &Path) -> Option<ImageAnnotation> {
    match fs::read_to_string(path) {
        Ok(content) => match serde_json::from_str::<ImageAnnotation>(&content) {
            Ok(annotation) => Some(annotation),
            Err(e) => {
                error!("Failed to parse JSON ({}): {:?}", path.display(), e);
                None
            }
        },
        Err(e) => {
            error!("Failed to read file ({}): {:?}", path.display(), e);
            None
        }
    }
}

fn main() {
    env_logger::init();
    let args = Args::parse();

    let dirname = PathBuf::from(&args.json_dir);

    // Check if args.json_dir exists
    if !dirname.exists() {
        error!("The specified json_dir does not exist: {}", args.json_dir);
        return;
    }

    info!("Starting the conversion process...");

    let pattern = dirname.join("**/*.json");
    let labels_dir = dirname.join("YOLODataset/labels");
    let images_dir = dirname.join("YOLODataset/images");
    create_dir(&labels_dir);
    create_dir(&images_dir);
    let train_labels_dir = labels_dir.join("train");
    let val_labels_dir = labels_dir.join("val");
    let train_images_dir = images_dir.join("train");
    let val_images_dir = images_dir.join("val");
    create_dir(&train_labels_dir);
    create_dir(&val_labels_dir);
    create_dir(&train_images_dir);
    create_dir(&val_images_dir);
    let (test_labels_dir, test_images_dir) = if args.test_size > 0.0 {
        let test_labels_dir = labels_dir.join("test");
        let test_images_dir = images_dir.join("test");
        create_dir(&test_labels_dir);
        create_dir(&test_images_dir);
        (Some(test_labels_dir), Some(test_images_dir))
    } else {
        (None, None)
    };
    let label_map = Arc::new(Mutex::new(HashMap::new()));
    let next_class_id = Arc::new(Mutex::new(0));
    let mut annotations = Vec::new();
    for entry in glob(pattern.to_str().expect("Failed to convert path to string"))
        .expect("Failed to read glob pattern")
    {
        if let Ok(path) = entry {
            if let Some(annotation) = read_and_parse_json(&path) {
                annotations.push((path, annotation));
            }
        }
    }

    info!("Read and parsed {} JSON files.", annotations.len());

    // Shuffle and split the annotations into train, val, and test sets
    let seed: u64 = 42; // Fixed random seed
    let mut rng = StdRng::seed_from_u64(seed);
    annotations.shuffle(&mut rng);
    let test_size = (annotations.len() as f32 * args.test_size).ceil() as usize;
    let val_size = (annotations.len() as f32 * args.val_size).ceil() as usize;
    let (test_annotations, rest_annotations) = annotations.split_at(test_size);
    let (val_annotations, train_annotations) = rest_annotations.split_at(val_size);

    info!(
        "Split data into {} training, {} validation, and {} test annotations.",
        train_annotations.len(),
        val_annotations.len(),
        test_annotations.len()
    );

    // Update label_map from label_list if not empty
    if !args.label_list.is_empty() {
        let mut label_map_guard = label_map.lock().unwrap();
        for (id, label) in args.label_list.iter().enumerate() {
            label_map_guard.insert(label.clone(), id);
        }
        *next_class_id.lock().unwrap() = args.label_list.len();
    }

    // Create progress bars
    let train_pb = create_progress_bar(train_annotations.len() as u64, "Train");
    let val_pb = create_progress_bar(val_annotations.len() as u64, "Val");
    let test_pb = create_progress_bar(test_annotations.len() as u64, "Test");

    // Process train_annotations in parallel
    info!("Processing training annotations...");
    process_annotations_in_parallel(
        &train_annotations,
        &train_labels_dir,
        &train_images_dir,
        &label_map,
        &next_class_id,
        &args,
        &dirname,
        &train_pb,
    );
    train_pb.finish_with_message("Train processing complete");

    // Process val_annotations in parallel
    info!("Processing validation annotations...");
    process_annotations_in_parallel(
        &val_annotations,
        &val_labels_dir,
        &val_images_dir,
        &label_map,
        &next_class_id,
        &args,
        &dirname,
        &val_pb,
    );
    val_pb.finish_with_message("Val processing complete");

    // Process test_annotations in parallel
    if let (Some(test_labels_dir), Some(test_images_dir)) = (test_labels_dir, test_images_dir) {
        info!("Processing test annotations...");
        process_annotations_in_parallel(
            &test_annotations,
            &test_labels_dir,
            &test_images_dir,
            &label_map,
            &next_class_id,
            &args,
            &dirname,
            &test_pb,
        );
        test_pb.finish_with_message("Test processing complete");
    }

    // Create dataset.yaml file after processing annotations
    info!("Creating dataset.yaml file...");
    create_dataset_yaml(&dirname, &args, &label_map);

    info!("Conversion process completed successfully.");
}

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

fn process_annotations_in_parallel(
    annotations: &[(PathBuf, ImageAnnotation)],
    labels_dir: &Path,
    images_dir: &Path,
    label_map: &Arc<Mutex<HashMap<String, usize>>>,
    next_class_id: &Arc<Mutex<usize>>,
    args: &Args,
    base_dir: &Path,
    pb: &ProgressBar,
) {
    annotations.par_iter().for_each(|(path, annotation)| {
        let mut label_map_guard = label_map.lock().unwrap();
        let mut next_class_id_guard = next_class_id.lock().unwrap();
        process_annotation(
            path,
            annotation,
            labels_dir,
            images_dir,
            &mut label_map_guard,
            &mut next_class_id_guard,
            args,
            base_dir,
        );
        pb.inc(1);
    });
}

fn create_dataset_yaml(
    dirname: &Path,
    args: &Args,
    label_map: &Arc<Mutex<HashMap<String, usize>>>,
) {
    let dataset_yaml_path = dirname.join("YOLODataset/dataset.yaml");
    let mut dataset_yaml =
        File::create(dataset_yaml_path).expect("Failed to create dataset.yaml file");
    let absolute_path =
        fs::canonicalize(&dirname.join("YOLODataset")).expect("Failed to get absolute path");
    let mut yaml_content = format!(
        "path: {}\ntrain: images/train\nval: images/val\n",
        absolute_path.to_str().unwrap()
    );
    if args.test_size > 0.0 {
        yaml_content.push_str("test: images/test\n");
    } else {
        yaml_content.push_str("test:\n");
    }
    yaml_content.push_str("\nnames:\n");
    // Read names from label_map
    let label_map_guard = label_map.lock().unwrap();
    let mut sorted_labels: Vec<_> = label_map_guard.iter().collect();
    sorted_labels.sort_by_key(|&(_, id)| id);
    for (label, id) in sorted_labels {
        yaml_content.push_str(&format!("    {}: {}\n", id, label));
    }
    dataset_yaml
        .write_all(yaml_content.as_bytes())
        .expect("Failed to write to dataset.yaml file");
}

fn process_annotation(
    path: &Path,
    annotation: &ImageAnnotation,
    labels_dir: &Path,
    images_dir: &Path,
    label_map: &mut HashMap<String, usize>,
    next_class_id: &mut usize,
    args: &Args,
    base_dir: &Path,
) {
    // Skip processing if image_path file does not exist and image_data is empty
    let image_path = base_dir.join(&annotation.image_path);
    if !image_path.exists()
        && annotation
            .image_data
            .as_ref()
            .map_or(true, |s| s.is_empty())
    {
        warn!(
            "Skipping annotation with non-existent image_path and empty image_data: {:?}",
            path
        );
        return;
    }

    let mut yolo_data = String::new();
    for shape in &annotation.shapes {
        let class_id = if args.label_list.is_empty() {
            // Dynamically generate class ID
            *label_map.entry(shape.label.clone()).or_insert_with(|| {
                let id = *next_class_id;
                *next_class_id += 1;
                id
            })
        } else {
            match args.label_list.iter().position(|r| r == &shape.label) {
                Some(id) => id,
                None => continue, // Ignore labels not in label_list
            }
        };

        if args.output_format == Format::Polygon {
            // Write polygon format
            yolo_data.push_str(&format!("{}", class_id));
            if shape.shape_type == "rectangle" {
                let (x1, y1) = shape.points[0];
                let (x2, y2) = shape.points[1];
                let rect_points = vec![(x1, y1), (x2, y1), (x2, y2), (x1, y2)];
                for &(x, y) in &rect_points {
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
            yolo_data.push_str("\n");
        } else {
            // Write bbox format
            let (x_min, y_min, x_max, y_max) = shape.points.iter().fold(
                (f64::MAX, f64::MAX, f64::MIN, f64::MIN),
                |(x_min, y_min, x_max, y_max), &(x, y)| {
                    (x_min.min(x), y_min.min(y), x_max.max(x), y_max.max(y))
                },
            );

            let x_center = (x_min + x_max) / 2.0 / annotation.image_width as f64;
            let y_center = (y_min + y_max) / 2.0 / annotation.image_height as f64;
            let width = (x_max - x_min) / annotation.image_width as f64;
            let height = (y_max - y_min) / annotation.image_height as f64;

            yolo_data.push_str(&format!(
                "{} {:.6} {:.6} {:.6} {:.6}\n",
                class_id, x_center, y_center, width, height
            ));
        }
    }

    let output_path = labels_dir
        .join(sanitize_filename::sanitize(
            path.file_stem().unwrap().to_str().unwrap(),
        ))
        .with_extension("txt");
    let mut file = File::create(output_path).expect("Failed to create YOLO data file");
    file.write_all(yolo_data.as_bytes())
        .expect("Failed to write YOLO data");

    // Copy image to images directory
    if image_path.exists() {
        let image_output_path = images_dir.join(sanitize_filename::sanitize(
            image_path.file_name().unwrap().to_str().unwrap(),
        ));
        copy(&image_path, &image_output_path).expect("Failed to copy image");
    } else if let Some(image_data) = &annotation.image_data {
        if !image_data.is_empty() {
            // Decode base64 image data and write to file
            let image_data = base64::decode(image_data).expect("Failed to decode image data");
            let extension = match image_path.extension().and_then(|ext| ext.to_str()) {
                Some(ext) => {
                    let ext_lower = ext.to_lowercase();
                    match ext_lower.as_str() {
                        "jpg" | "jpeg" => "jpeg",
                        _ => "png",
                    }
                }
                None => "png",
            };
            let image_output_path = images_dir
                .join(sanitize_filename::sanitize(
                    Path::new(&annotation.image_path)
                        .file_stem()
                        .unwrap()
                        .to_str()
                        .unwrap(),
                ))
                .with_extension(extension);
            let mut file = File::create(&image_output_path).expect("Failed to create image file");
            file.write_all(&image_data)
                .expect("Failed to write image data");
        }
    } else {
        warn!(
            "Image file not found and image data is empty: {:?}",
            image_path
        );
    }
}
