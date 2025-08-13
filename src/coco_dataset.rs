//! COCO dataset processing module
//!
//! This module provides functionality to process LabelMe annotations and convert them
//! to COCO format for instance segmentation and object detection.

use dashmap::{DashMap, DashSet};
use log::{debug, info, warn};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicUsize, Ordering::Relaxed},
    Arc, Mutex,
};

use crate::coco::{Annotation, CocoConfig, CocoFile, Image};
use crate::config::Args;
use crate::types::{ImageAnnotation, Shape};
use crate::utils::{
    create_io_thread_pool, create_output_directory, infer_image_format, read_and_parse_json,
    read_and_parse_json_buffered, read_and_parse_json_streaming,
};

/// Struct to hold the paths to the output directories for COCO dataset
#[derive(Debug)]
pub struct CocoOutputDirs {
    pub annotations_dir: PathBuf,
    pub train_images_dir: PathBuf,
    pub val_images_dir: PathBuf,
    pub test_images_dir: Option<PathBuf>,
}

/// Struct to hold COCO data for a split
#[derive(Debug)]
pub struct CocoSplitData {
    pub images: Vec<Image>,
    pub annotations: Vec<Annotation>,
}

/// Type alias for the complex return type of process_json_files_for_coco
type ProcessJsonFilesResult = Result<
    (
        CocoSplitData,
        CocoSplitData,
        CocoSplitData,
        std::collections::HashSet<String>,
    ),
    Box<dyn std::error::Error>,
>;

/// Type alias for the complex return type of process_background_images_for_coco
type ProcessBackgroundImagesResult =
    Result<(Vec<Image>, Vec<Image>, Vec<Image>), Box<dyn std::error::Error>>;

/// Parameters for processing JSON files for COCO conversion
#[derive(Debug)]
struct ProcessJsonFilesParams<'a> {
    dirname: &'a Path,
    args: &'a Args,
    output_dirs: &'a CocoOutputDirs,
    label_map: &'a DashMap<String, usize>,
    next_class_id: &'a Arc<AtomicUsize>,
    coco_config: &'a CocoConfig,
    image_id_counter: &'a Arc<AtomicUsize>,
    annotation_id_counter: &'a Arc<AtomicUsize>,
}

/// Parameters for processing a single annotation for COCO conversion
#[derive(Debug)]
struct ProcessAnnotationParams<'a> {
    json_path: &'a Path,
    annotation: &'a ImageAnnotation,
    args: &'a Args,
    output_dirs: &'a CocoOutputDirs,
    label_map: &'a DashMap<String, usize>,
    next_class_id: &'a Arc<AtomicUsize>,
    coco_config: &'a CocoConfig,
    processed_image_basenames: &'a DashSet<String>,
    base_dir: &'a Path,
    train_data: &'a Arc<Mutex<CocoSplitData>>,
    val_data: &'a Arc<Mutex<CocoSplitData>>,
    test_data: &'a Arc<Mutex<CocoSplitData>>,
    image_id_counter: &'a Arc<AtomicUsize>,
    annotation_id_counter: &'a Arc<AtomicUsize>,
}

/// Parameters for writing COCO files
#[derive(Debug)]
struct WriteCocoFilesParams<'a> {
    annotations_dir: &'a Path,
    train_data: CocoSplitData,
    val_data: CocoSplitData,
    test_data: CocoSplitData,
    train_bg_images: Vec<Image>,
    val_bg_images: Vec<Image>,
    test_bg_images: Vec<Image>,
    label_map: &'a DashMap<String, usize>,
    has_test_split: bool,
}

/// Set up the directory structure for COCO dataset output
pub fn setup_coco_output_directories(
    args: &Args,
    dirname: &Path,
) -> std::io::Result<CocoOutputDirs> {
    let base_dir = dirname.join("COCODataset");
    let annotations_dir = create_output_directory(&base_dir.join("annotations"))?;
    let images_dir = create_output_directory(&base_dir.join("images"))?;

    let train_images_dir = create_output_directory(&images_dir.join("train"))?;
    let val_images_dir = create_output_directory(&images_dir.join("val"))?;

    let test_images_dir = if args.test_size > 0.0 {
        Some(create_output_directory(&images_dir.join("test"))?)
    } else {
        None
    };

    Ok(CocoOutputDirs {
        annotations_dir,
        train_images_dir,
        val_images_dir,
        test_images_dir,
    })
}

/// Main COCO dataset processing pipeline
pub fn process_coco_dataset(
    output_dirs: &CocoOutputDirs,
    args: &Args,
    dirname: &Path,
    coco_config: &CocoConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    // Use a local HashMap for initialization to avoid DashMap overhead
    let mut label_map_local = HashMap::new();
    let next_class_id = Arc::new(AtomicUsize::new(0));

    // Initialize the label map with labels from label_list if specified
    if !args.label_list.is_empty() {
        for (id, label) in args.label_list.iter().enumerate() {
            label_map_local.insert(label.clone(), id);
        }
        next_class_id.store(args.label_list.len(), Relaxed);
    }

    // Convert to DashMap for concurrent access in later stages
    let label_map: DashMap<String, usize> = label_map_local.into_iter().collect();

    // Initialize ID counters for deterministic assignment
    let image_id_counter = Arc::new(AtomicUsize::new(coco_config.start_image_id as usize));
    let annotation_id_counter =
        Arc::new(AtomicUsize::new(coco_config.start_annotation_id as usize));

    // Process JSON files
    info!("Processing JSON files for COCO conversion...");
    let (train_data, val_data, test_data, processed_image_basenames) =
        process_json_files_for_coco(ProcessJsonFilesParams {
            dirname,
            args,
            output_dirs,
            label_map: &label_map,
            next_class_id: &next_class_id,
            coco_config,
            image_id_counter: &image_id_counter,
            annotation_id_counter: &annotation_id_counter,
        })?;

    // Process background images
    info!("Processing background images for COCO conversion...");
    let (train_bg_images, val_bg_images, test_bg_images) = process_background_images_for_coco(
        dirname,
        args,
        output_dirs,
        &processed_image_basenames,
        &image_id_counter,
    )?;

    // Write COCO JSON files
    info!("Writing COCO JSON files...");
    write_coco_files(WriteCocoFilesParams {
        annotations_dir: &output_dirs.annotations_dir,
        train_data,
        val_data,
        test_data,
        train_bg_images,
        val_bg_images,
        test_bg_images,
        label_map: &label_map,
        has_test_split: args.test_size > 0.0,
    })?;

    info!("COCO conversion process completed successfully.");
    Ok(())
}

/// Process JSON files for COCO conversion
fn process_json_files_for_coco(params: ProcessJsonFilesParams) -> ProcessJsonFilesResult {
    let ProcessJsonFilesParams {
        dirname,
        args,
        output_dirs,
        label_map,
        next_class_id,
        coco_config,
        image_id_counter,
        annotation_id_counter,
    } = params;
    // Create a custom thread pool with limited concurrency
    let thread_pool = create_io_thread_pool(args.workers);

    // Walk through the directory structure to find JSON files
    use jwalk::WalkDir;
    use rayon::prelude::*;
    use std::sync::Mutex;

    let json_entries = WalkDir::new(dirname)
        .skip_hidden(false)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            // Skip COCO Dataset directories early by comparing directory names directly
            if e.file_type().is_dir() {
                if let Some(name) = e.file_name().to_str() {
                    name != "COCODataset"
                } else {
                    false
                }
            } else {
                true
            }
        })
        .filter(|e| {
            e.file_type().is_file() && e.path().extension().is_some_and(|ext| ext == "json")
        })
        .map(|e| e.path().to_path_buf())
        .par_bridge();

    // Create counters for tracking processed files
    let processed_count = std::sync::atomic::AtomicUsize::new(0);
    let message_update_count = std::sync::atomic::AtomicUsize::new(0);
    const MESSAGE_UPDATE_INTERVAL: usize = 100;

    // Create an indeterminate progress bar
    let pb = indicatif::ProgressBar::new_spinner();
    pb.set_style(
        indicatif::ProgressStyle::default_spinner()
            .template("{spinner:.green} [{elapsed_precise}] [Processing JSON files for COCO] {msg}")
            .progress_chars("#>-"),
    );
    pb.enable_steady_tick(100);
    pb.set_message("Processing files...");

    // Use thread-safe collections to track processed data
    let processed_image_basenames: DashSet<String> = DashSet::new();
    let train_data = Arc::new(Mutex::new(CocoSplitData {
        images: Vec::new(),
        annotations: Vec::new(),
    }));
    let val_data = Arc::new(Mutex::new(CocoSplitData {
        images: Vec::new(),
        annotations: Vec::new(),
    }));
    let test_data = Arc::new(Mutex::new(CocoSplitData {
        images: Vec::new(),
        annotations: Vec::new(),
    }));

    // Process JSON files in parallel
    thread_pool.install(|| {
        json_entries.for_each(|json_path| {
            // Try to parse with simd-json for faster performance
            if let Some(annotation) = read_and_parse_json_buffered(&json_path, args.buffer_size_kib)
            {
                process_annotation_for_coco(ProcessAnnotationParams {
                    json_path: &json_path,
                    annotation: &annotation,
                    args,
                    output_dirs,
                    label_map,
                    next_class_id,
                    coco_config,
                    processed_image_basenames: &processed_image_basenames,
                    base_dir: dirname,
                    train_data: &train_data,
                    val_data: &val_data,
                    test_data: &test_data,
                    image_id_counter,
                    annotation_id_counter,
                });
            } else if let Some(mut streaming_annotation) =
                read_and_parse_json_streaming(&json_path, args.buffer_size_kib)
            {
                // Convert streaming annotation to regular annotation
                if let Ok(regular_annotation) = streaming_annotation.to_image_annotation() {
                    process_annotation_for_coco(ProcessAnnotationParams {
                        json_path: &json_path,
                        annotation: &regular_annotation,
                        args,
                        output_dirs,
                        label_map,
                        next_class_id,
                        coco_config,
                        processed_image_basenames: &processed_image_basenames,
                        base_dir: dirname,
                        train_data: &train_data,
                        val_data: &val_data,
                        test_data: &test_data,
                        image_id_counter,
                        annotation_id_counter,
                    });
                }
            } else if let Some(annotation) = read_and_parse_json(&json_path, args.buffer_size_kib) {
                process_annotation_for_coco(ProcessAnnotationParams {
                    json_path: &json_path,
                    annotation: &annotation,
                    args,
                    output_dirs,
                    label_map,
                    next_class_id,
                    coco_config,
                    processed_image_basenames: &processed_image_basenames,
                    base_dir: dirname,
                    train_data: &train_data,
                    val_data: &val_data,
                    test_data: &test_data,
                    image_id_counter,
                    annotation_id_counter,
                });
            }

            // Update the progress counter
            pb.inc(1);

            // Update message periodically
            let count = processed_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            let update_counter =
                message_update_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if update_counter % MESSAGE_UPDATE_INTERVAL == 0 {
                pb.set_message(format!("Processed {} files...", count));
            }
        });
    });

    // Finish the progress bar
    pb.finish();

    // Extract the data from the mutexes
    let train_final = match Arc::try_unwrap(train_data) {
        Ok(mutex) => match mutex.into_inner() {
            Ok(inner) => inner,
            Err(poisoned) => {
                warn!("train_data mutex poisoned at finalize; using inner value");
                poisoned.into_inner()
            }
        },
        Err(arc) => match arc.lock() {
            Ok(mut guard) => {
                use std::mem::take;
                CocoSplitData {
                    images: take(&mut guard.images),
                    annotations: take(&mut guard.annotations),
                }
            }
            Err(poisoned) => {
                warn!("train_data mutex poisoned and Arc not unique; using inner");
                let mut guard = poisoned.into_inner();
                use std::mem::take;
                CocoSplitData {
                    images: take(&mut guard.images),
                    annotations: take(&mut guard.annotations),
                }
            }
        },
    };
    let val_final = match Arc::try_unwrap(val_data) {
        Ok(mutex) => match mutex.into_inner() {
            Ok(inner) => inner,
            Err(poisoned) => {
                warn!("val_data mutex poisoned at finalize; using inner value");
                poisoned.into_inner()
            }
        },
        Err(arc) => match arc.lock() {
            Ok(mut guard) => {
                use std::mem::take;
                CocoSplitData {
                    images: take(&mut guard.images),
                    annotations: take(&mut guard.annotations),
                }
            }
            Err(poisoned) => {
                warn!("val_data mutex poisoned and Arc not unique; using inner");
                let mut guard = poisoned.into_inner();
                use std::mem::take;
                CocoSplitData {
                    images: take(&mut guard.images),
                    annotations: take(&mut guard.annotations),
                }
            }
        },
    };
    let test_final = match Arc::try_unwrap(test_data) {
        Ok(mutex) => match mutex.into_inner() {
            Ok(inner) => inner,
            Err(poisoned) => {
                warn!("test_data mutex poisoned at finalize; using inner value");
                poisoned.into_inner()
            }
        },
        Err(arc) => match arc.lock() {
            Ok(mut guard) => {
                use std::mem::take;
                CocoSplitData {
                    images: take(&mut guard.images),
                    annotations: take(&mut guard.annotations),
                }
            }
            Err(poisoned) => {
                warn!("test_data mutex poisoned and Arc not unique; using inner");
                let mut guard = poisoned.into_inner();
                use std::mem::take;
                CocoSplitData {
                    images: take(&mut guard.images),
                    annotations: take(&mut guard.annotations),
                }
            }
        },
    };

    // Convert the DashSet to a regular HashSet
    Ok((
        train_final,
        val_final,
        test_final,
        processed_image_basenames.into_iter().collect(),
    ))
}

/// Process a single annotation for COCO conversion
fn process_annotation_for_coco(params: ProcessAnnotationParams) {
    let ProcessAnnotationParams {
        json_path,
        annotation,
        args,
        output_dirs,
        label_map,
        next_class_id,
        coco_config,
        processed_image_basenames,
        base_dir,
        train_data,
        val_data,
        test_data,
        image_id_counter,
        annotation_id_counter,
    } = params;
    // Determine the image path relative to the JSON file's directory
    let image_path = json_path
        .parent()
        .map(|parent| parent.join(&annotation.image_path))
        .unwrap_or_else(|| base_dir.join(&annotation.image_path));

    // Add labels to the label map if not using a predefined list and not in deterministic mode
    if args.label_list.is_empty() && !args.deterministic_labels {
        let unique_labels: std::collections::HashSet<&str> = annotation
            .shapes
            .iter()
            .map(|shape| shape.label.as_str())
            .collect();

        for label in unique_labels {
            label_map.entry(label.to_string()).or_insert_with(|| {
                next_class_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            });
        }
    }

    // Determine which split this image belongs to
    let (data_mutex, images_dir) = {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        image_path.hash(&mut hasher);
        let hash = hasher.finish();
        let ratio = (hash % 1000) as f32 / 1000.0;

        if ratio < args.val_size {
            // Validation set
            (val_data, &output_dirs.val_images_dir)
        } else if ratio < args.val_size + args.test_size {
            // Test set
            if let Some(test_images_dir) = &output_dirs.test_images_dir {
                (test_data, test_images_dir)
            } else {
                (train_data, &output_dirs.train_images_dir)
            }
        } else {
            // Training set
            (train_data, &output_dirs.train_images_dir)
        }
    };

    // Copy the image file with embedded image data support
    if let Err(e) =
        copy_image_for_coco_with_data(&image_path, images_dir, annotation.image_data.as_ref())
    {
        warn!("Failed to copy image {}: {}", image_path.display(), e);
        return;
    }

    // Get the file name for the image
    let file_name = image_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    // Create image entry
    let image = Image::new(
        0, // ID will be assigned later
        file_name.to_string(),
        annotation.image_width,
        annotation.image_height,
    );

    // Process shapes
    let mut annotations = Vec::new();
    for shape in &annotation.shapes {
        if let Some(class_id) = label_map.get(&shape.label) {
            if let Ok(ann) = convert_shape_to_coco_annotation(
                shape,
                0, // image_id will be assigned later
                *class_id as u32,
                annotation.image_width,
                annotation.image_height,
                coco_config.segmentation_mode,
            ) {
                annotations.push(ann);
            }
        } else {
            debug!("Skipping shape with unknown label: {}", shape.label);
        }
    }

    // Add to the appropriate split data
    if let Ok(mut data) = data_mutex.lock() {
        // Get deterministic IDs from counters
        let image_id = image_id_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed) as u32;

        // Update image ID
        let mut image_with_id = image.clone();
        image_with_id.id = image_id;
        data.images.push(image_with_id);

        // Update annotation image IDs
        for mut ann in annotations {
            ann.image_id = image_id;
            ann.id =
                annotation_id_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed) as u32;
            data.annotations.push(ann);
        }
    }

    // Track processed image basenames
    processed_image_basenames.insert(file_name.to_string());
}

/// Copy image file for COCO dataset with embedded image data support
fn copy_image_for_coco_with_data(
    image_path: &Path,
    images_dir: &Path,
    image_data: Option<&String>,
) -> std::io::Result<()> {
    if image_path.exists() {
        let file_name = image_path
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Invalid file name: {:?}", image_path),
                )
            })?;

        let dest_path = images_dir.join(file_name);
        fs::copy(image_path, dest_path)?;
    } else if let Some(image_data) = image_data {
        if !image_data.is_empty() {
            // Handle missing image file by extracting image_data from JSON
            let file_name = image_path
                .file_name()
                .and_then(|s| s.to_str())
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Invalid file name: {:?}", image_path),
                    )
                })?;

            // Infer image format by decoding just the header
            let image_extension = {
                // Create a cursor to read from the base64 string
                let mut cursor = std::io::Cursor::new(image_data);

                // Create a base64 decoder that reads from the cursor
                let mut decoder = base64::read::DecoderReader::new(&mut cursor, base64::STANDARD);

                // Read first 16 bytes to determine image format
                let mut header_buffer = [0u8; 16];
                let bytes_read = std::io::Read::read(&mut decoder, &mut header_buffer)?;

                // Infer image format from the header
                infer_image_format(&header_buffer[..bytes_read]).unwrap_or("png")
            };

            // Create the output file with the correct extension
            let dest_path = images_dir.join(file_name).with_extension(image_extension);

            // Decode the full data and stream it to the file
            let mut cursor = std::io::Cursor::new(image_data);
            let mut decoder = base64::read::DecoderReader::new(&mut cursor, base64::STANDARD);

            let mut file = File::create(&dest_path)?;

            // Copy the decoded data directly to the file (streaming)
            std::io::copy(&mut decoder, &mut file)?;
        } else {
            // No image data available
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "Image file not found and no image data available: {:?}",
                    image_path
                ),
            ));
        }
    } else {
        // No image file or image data available
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "Image file not found and no image data available: {:?}",
                image_path
            ),
        ));
    }
    Ok(())
}

/// Convert a LabelMe shape to a COCO annotation
fn convert_shape_to_coco_annotation(
    shape: &Shape,
    image_id: u32,
    category_id: u32,
    _image_width: u32,
    _image_height: u32,
    segmentation_mode: crate::config::SegmentationMode,
) -> std::io::Result<Annotation> {
    let (bbox, area, segmentation) = match shape.shape_type.as_str() {
        "polygon" => {
            let mut polygon_points = shape.points.clone();

            // Remove duplicate last point if present
            if polygon_points.len() >= 4 {
                let first = polygon_points[0];
                let last = polygon_points[polygon_points.len() - 1];
                if first == last {
                    polygon_points.pop();
                }
            }

            // Convert to flat vector
            let flat_polygon: Vec<f64> = polygon_points
                .into_iter()
                .flat_map(|(x, y)| [x, y])
                .collect();

            let area = crate::coco::calculate_polygon_area(&flat_polygon);
            let bbox = crate::coco::calculate_bbox_from_polygon(&flat_polygon);

            let segmentation = match segmentation_mode {
                crate::config::SegmentationMode::Polygon => Some(vec![flat_polygon]),
                crate::config::SegmentationMode::BboxOnly => None,
            };

            (bbox, area, segmentation)
        }
        "rectangle" => {
            if shape.points.len() < 2 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Rectangle must have at least 2 points",
                ));
            }

            let (x1, y1) = shape.points[0];
            let (x2, y2) = shape.points[1];
            let polygon_points = crate::coco::rectangle_to_polygon(x1, y1, x2, y2);

            let area = crate::coco::calculate_polygon_area(&polygon_points);
            let bbox = crate::coco::calculate_bbox_from_polygon(&polygon_points);

            let segmentation = match segmentation_mode {
                crate::config::SegmentationMode::Polygon => Some(vec![polygon_points]),
                crate::config::SegmentationMode::BboxOnly => None,
            };

            (bbox, area, segmentation)
        }
        "circle" => {
            if shape.points.len() < 2 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Circle must have at least 2 points",
                ));
            }

            let (cx, cy) = shape.points[0];
            let (px, py) = shape.points[1];
            let radius = ((cx - px).powi(2) + (cy - py).powi(2)).sqrt();
            let polygon_points = crate::coco::circle_to_polygon(cx, cy, radius, 12);

            let area = crate::coco::calculate_polygon_area(&polygon_points);
            let bbox = crate::coco::calculate_bbox_from_polygon(&polygon_points);

            let segmentation = match segmentation_mode {
                crate::config::SegmentationMode::Polygon => Some(vec![polygon_points]),
                crate::config::SegmentationMode::BboxOnly => None,
            };

            (bbox, area, segmentation)
        }
        _ => {
            // Skip unsupported shape types
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unsupported shape type: {}", shape.shape_type),
            ));
        }
    };

    // Skip zero-area shapes
    if area <= 0.0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Shape has zero area",
        ));
    }

    Ok(Annotation::new(
        0,
        image_id,
        category_id,
        bbox,
        area,
        0,
        segmentation,
    ))
}

/// Process background images for COCO conversion
fn process_background_images_for_coco(
    dirname: &Path,
    args: &Args,
    output_dirs: &CocoOutputDirs,
    processed_image_basenames: &std::collections::HashSet<String>,
    image_id_counter: &Arc<AtomicUsize>,
) -> ProcessBackgroundImagesResult {
    // If include_background is not set, we don't need to process background images
    if !args.include_background {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }

    // Create a custom thread pool with limited concurrency
    let thread_pool = create_io_thread_pool(args.workers);

    // Use the precomputed set of supported image extensions for fast lookup
    let image_extensions = crate::types::get_image_extensions_set();

    // Walk through the directory structure to find image files
    use jwalk::WalkDir;
    use rayon::prelude::*;
    use std::sync::Mutex;

    let image_entries = WalkDir::new(dirname)
        .skip_hidden(false)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            // Skip COCO Dataset directories early by comparing directory names directly
            if e.file_type().is_dir() {
                if let Some(name) = e.file_name().to_str() {
                    name != "COCODataset"
                } else {
                    false
                }
            } else {
                true
            }
        })
        .filter(|e| e.file_type().is_file())
        .filter_map(|e| {
            let path = e.path();
            if let Some(extension) = path.extension() {
                let ext = extension.to_string_lossy().to_lowercase();
                if image_extensions.contains(&ext) {
                    return Some(path.to_path_buf());
                }
            }
            None
        })
        .par_bridge();

    // Create a progress bar for background image processing
    let pb = indicatif::ProgressBar::new_spinner();
    pb.set_style(
        indicatif::ProgressStyle::default_spinner()
            .template("{spinner:.green} [{elapsed_precise}] [Processing background images for COCO] {msg}")
            .progress_chars("#>-"),
    );
    pb.enable_steady_tick(100);
    pb.set_message("Processing background images...");

    // Use thread-safe collections to track background images
    let train_bg_images = Arc::new(Mutex::new(Vec::new()));
    let val_bg_images = Arc::new(Mutex::new(Vec::new()));
    let test_bg_images = Arc::new(Mutex::new(Vec::new()));

    // Process background images in parallel
    thread_pool.install(|| {
        image_entries.for_each(|image_path| {
            let file_name = image_path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");

            // Check if this image was already processed (has a JSON annotation)
            if !processed_image_basenames.contains(file_name) {
                // This is a background image, process it
                let (bg_images, images_dir) = {
                    use std::collections::hash_map::DefaultHasher;
                    use std::hash::{Hash, Hasher};
                    let mut hasher = DefaultHasher::new();
                    image_path.hash(&mut hasher);
                    let hash = hasher.finish();
                    let ratio = (hash % 1000) as f32 / 1000.0;

                    if ratio < args.val_size {
                        // Validation set
                        (val_bg_images.clone(), &output_dirs.val_images_dir)
                    } else if ratio < args.val_size + args.test_size {
                        // Test set
                        if let Some(test_images_dir) = &output_dirs.test_images_dir {
                            (test_bg_images.clone(), test_images_dir)
                        } else {
                            (train_bg_images.clone(), &output_dirs.train_images_dir)
                        }
                    } else {
                        // Training set
                        (train_bg_images.clone(), &output_dirs.train_images_dir)
                    }
                };

                // Copy the image file
                if let Err(e) = copy_image_for_coco_with_data(&image_path, images_dir, None) {
                    warn!(
                        "Failed to copy background image {}: {}",
                        image_path.display(),
                        e
                    );
                } else {
                    // Read actual image dimensions from the image file header
                    let (image_width, image_height) = match crate::utils::read_image_dimensions(&image_path) {
                        Ok(dimensions) => dimensions,
                        Err(e) => {
                            warn!(
                                "Failed to read image dimensions for background image {}: {}. Using fallback dimensions.",
                                image_path.display(),
                                e
                            );
                            // Fallback to reasonable default dimensions if we can't read the actual ones
                            (1024, 1024)
                        }
                    };

                    // Create image entry
                    let image = Image::new(0, file_name.to_string(), image_width, image_height);

                    // Add to the appropriate split
                    if let Ok(mut images) = bg_images.lock() {
                        // Get deterministic ID from counter
                        let image_id = image_id_counter
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                            as u32;
                        let mut image_with_id = image;
                        image_with_id.id = image_id;
                        images.push(image_with_id);
                    }
                }

                // Update the progress counter
                pb.inc(1);
            }
        });
    });

    // Finish the progress bar
    pb.finish();

    // Extract the data from the mutexes
    let train_final = match Arc::try_unwrap(train_bg_images) {
        Ok(mutex) => match mutex.into_inner() {
            Ok(inner) => inner,
            Err(poisoned) => {
                warn!("train_bg_images mutex poisoned at finalize; using inner value");
                poisoned.into_inner()
            }
        },
        Err(arc) => match arc.lock() {
            Ok(mut guard) => {
                use std::mem::take;
                take(&mut *guard)
            }
            Err(poisoned) => {
                warn!("train_bg_images mutex poisoned and Arc not unique; using inner");
                let mut guard = poisoned.into_inner();
                use std::mem::take;
                take(&mut *guard)
            }
        },
    };
    let val_final = match Arc::try_unwrap(val_bg_images) {
        Ok(mutex) => match mutex.into_inner() {
            Ok(inner) => inner,
            Err(poisoned) => {
                warn!("val_bg_images mutex poisoned at finalize; using inner value");
                poisoned.into_inner()
            }
        },
        Err(arc) => match arc.lock() {
            Ok(mut guard) => {
                use std::mem::take;
                take(&mut *guard)
            }
            Err(poisoned) => {
                warn!("val_bg_images mutex poisoned and Arc not unique; using inner");
                let mut guard = poisoned.into_inner();
                use std::mem::take;
                take(&mut *guard)
            }
        },
    };
    let test_final = match Arc::try_unwrap(test_bg_images) {
        Ok(mutex) => match mutex.into_inner() {
            Ok(inner) => inner,
            Err(poisoned) => {
                warn!("test_bg_images mutex poisoned at finalize; using inner value");
                poisoned.into_inner()
            }
        },
        Err(arc) => match arc.lock() {
            Ok(mut guard) => {
                use std::mem::take;
                take(&mut *guard)
            }
            Err(poisoned) => {
                warn!("test_bg_images mutex poisoned and Arc not unique; using inner");
                let mut guard = poisoned.into_inner();
                use std::mem::take;
                take(&mut *guard)
            }
        },
    };

    Ok((train_final, val_final, test_final))
}

/// Write COCO JSON files for each split
fn write_coco_files(params: WriteCocoFilesParams) -> std::io::Result<()> {
    let WriteCocoFilesParams {
        annotations_dir,
        train_data,
        val_data,
        test_data,
        train_bg_images,
        val_bg_images,
        test_bg_images,
        label_map,
        has_test_split,
    } = params;
    // Create categories
    let mut categories: Vec<crate::coco::Category> = label_map
        .iter()
        .map(|entry| crate::coco::Category {
            id: (*entry.value() + 1) as u32,
            name: entry.key().clone(),
            supercategory: "none".to_string(),
        })
        .collect();

    // Sort by ID to ensure consistent ordering
    categories.sort_by_key(|c| c.id);

    // Combine regular images with background images
    let train_images: Vec<Image> = train_data
        .images
        .into_iter()
        .chain(train_bg_images)
        .collect();

    let val_images: Vec<Image> = val_data.images.into_iter().chain(val_bg_images).collect();

    let test_images: Vec<Image> = test_data.images.into_iter().chain(test_bg_images).collect();

    // Create COCO files
    let train_coco_file = CocoFile {
        info: crate::coco::Info::default(),
        licenses: vec![crate::coco::License::default()],
        categories: categories.clone(),
        images: train_images,
        annotations: train_data.annotations,
    };

    let val_coco_file = CocoFile {
        info: crate::coco::Info::default(),
        licenses: vec![crate::coco::License::default()],
        categories: categories.clone(),
        images: val_images,
        annotations: val_data.annotations,
    };

    let test_coco_file = CocoFile {
        info: crate::coco::Info::default(),
        licenses: vec![crate::coco::License::default()],
        categories,
        images: test_images,
        annotations: test_data.annotations,
    };

    // Write train file
    let train_path = annotations_dir.join("instances_train.json");
    let train_file = File::create(&train_path)?;
    let mut train_writer = BufWriter::new(train_file);
    serde_json::to_writer(&mut train_writer, &train_coco_file)?;
    info!("Wrote {}", train_path.display());

    // Write validation file
    let val_path = annotations_dir.join("instances_val.json");
    let val_file = File::create(&val_path)?;
    let mut val_writer = BufWriter::new(val_file);
    serde_json::to_writer(&mut val_writer, &val_coco_file)?;
    info!("Wrote {}", val_path.display());

    // Write test file if needed
    if has_test_split {
        let test_path = annotations_dir.join("instances_test.json");
        let test_file = File::create(&test_path)?;
        let mut test_writer = BufWriter::new(test_file);
        serde_json::to_writer(&mut test_writer, &test_coco_file)?;
        info!("Wrote {}", test_path.display());
    }

    Ok(())
}
