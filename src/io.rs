use dashmap::DashMap;
use jwalk::WalkDir;
use rayon::prelude::*;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

use crate::config::Args;
use crate::types::{OutputDirs, SplitData};
use crate::utils::{
    create_io_thread_pool, create_output_directory, read_and_parse_json,
    read_and_parse_json_buffered, read_and_parse_json_streaming,
};

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

/// Process JSON files in a streaming fashion to reduce memory footprint
/// Returns a set of processed image basenames for background image detection and processing statistics
pub fn process_json_files_streaming(
    dirname: &Path,
    args: &Args,
    output_dirs: &OutputDirs,
    label_map: &dashmap::DashMap<String, usize>,
    next_class_id: &std::sync::Arc<std::sync::atomic::AtomicUsize>,
    _split_data: &mut SplitData,
) -> (
    std::collections::HashSet<String>,
    crate::types::ProcessingStats,
) {
    // Create a custom thread pool with limited concurrency
    let thread_pool = create_io_thread_pool(args.workers);

    // Walk through the directory structure to find JSON files using parallel jwalk
    let json_entries = WalkDir::new(dirname)
        .skip_hidden(false)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            // Skip YOLODataset directories early by comparing directory names directly
            if e.file_type().is_dir() {
                if let Some(name) = e.file_name().to_str() {
                    name != "YOLODataset"
                } else {
                    // If we can't convert to str, skip to be safe
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

    // Create a counter for tracking processed files
    let processed_count = std::sync::atomic::AtomicUsize::new(0);
    // Counter for periodic message updates (every 100 files)
    let message_update_count = std::sync::atomic::AtomicUsize::new(0);
    const MESSAGE_UPDATE_INTERVAL: usize = 100;

    // Create an indeterminate progress bar since we don't know the exact count upfront
    let pb = indicatif::ProgressBar::new_spinner();
    pb.set_style(
        indicatif::ProgressStyle::default_spinner()
            .template("{spinner:.green} [{elapsed_precise}] [Processing JSON files] {msg}")
            .progress_chars("#>-"),
    );
    pb.enable_steady_tick(100);
    pb.set_message("Processing files...");

    // Use a thread-safe collection to track processed image basenames
    let processed_image_basenames: dashmap::DashSet<String> = dashmap::DashSet::new();

    // Create a cache for sanitized filenames to avoid recomputing
    let filename_cache: Arc<DashMap<String, String>> = Arc::new(DashMap::new());

    // Create thread-safe statistics collection
    let stats = Arc::new(std::sync::Mutex::new(crate::types::ProcessingStats::new()));

    // Process JSON files in parallel using the custom thread pool
    thread_pool.install(|| {
        json_entries.for_each(|json_path| {
            // Increment total files processed counter
            if let Ok(mut stats) = stats.lock() {
                stats.increment_total();
            }

            // First, try to parse with simd-json for faster performance
            if let Some(annotation) = read_and_parse_json_buffered(&json_path, args.buffer_size_kib)
            {
                // Determine the image path relative to the JSON file's directory
                let image_path = json_path
                    .parent()
                    .map(|parent| parent.join(&annotation.image_path))
                    .unwrap_or_else(|| dirname.join(&annotation.image_path));

                // Add labels to the label map if not using a predefined list and not in deterministic mode
                if args.label_list.is_empty() && !args.deterministic_labels {
                    // Collect unique labels from this annotation to avoid repeated cloning and contention
                    let unique_labels: std::collections::HashSet<&str> = annotation
                        .shapes
                        .iter()
                        .map(|shape| shape.label.as_str())
                        .collect();

                    // Insert each unique label once per file
                    for label in unique_labels {
                        label_map.entry(label.to_string()).or_insert_with(|| {
                            next_class_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                        });
                    }
                }

                // Process the annotation immediately (copy image, generate label file)
                // Distribute files according to split ratios using a hash-based approach
                let (labels_dir, images_dir) = {
                    // Use a hash of the image path for consistent distribution
                    let hash = {
                        use std::collections::hash_map::DefaultHasher;
                        use std::hash::{Hash, Hasher};
                        let mut hasher = DefaultHasher::new();
                        image_path.hash(&mut hasher);
                        hasher.finish()
                    };
                    let ratio = (hash % 1000) as f32 / 1000.0;

                    if ratio < args.val_size {
                        // Validation set
                        (&output_dirs.val_labels_dir, &output_dirs.val_images_dir)
                    } else if ratio < args.val_size + args.test_size {
                        // Test set
                        (
                            output_dirs
                                .test_labels_dir
                                .as_ref()
                                .unwrap_or(&output_dirs.train_labels_dir),
                            output_dirs
                                .test_images_dir
                                .as_ref()
                                .unwrap_or(&output_dirs.train_images_dir),
                        )
                    } else {
                        // Training set
                        (&output_dirs.train_labels_dir, &output_dirs.train_images_dir)
                    }
                };

                let processor = crate::conversion::AnnotationProcessor {
                    image_path: &image_path,
                    annotation: Some(&annotation),
                    streaming_annotation: None,
                    labels_dir,
                    images_dir,
                    label_map,
                    args,
                    filename_cache: &filename_cache,
                    base_dir: dirname,
                    stats: Some(&mut stats.lock().unwrap()),
                };
                if let Err(e) = crate::conversion::process_annotation(processor) {
                    log::error!(
                        "Failed to process annotation {}: {}",
                        json_path.display(),
                        e
                    );
                }

                // Track processed image basenames for background image detection
                // Use the relative path as the cache key
                let relative_path = image_path
                    .strip_prefix(dirname)
                    .unwrap_or(image_path.as_path());
                let cache_key = relative_path.to_string_lossy().to_string();

                // Use the sanitized name for collision-free tracking
                if let Some(file_stem) = image_path.file_stem().and_then(|s| s.to_str()) {
                    let sanitized_name = if let Some(cached) = filename_cache.get(&cache_key) {
                        cached.clone()
                    } else {
                        // Generate a collision-resistant name using the file stem and relative path
                        let collision_resistant_name =
                            crate::utils::generate_collision_resistant_name(
                                file_stem,
                                relative_path,
                            );
                        filename_cache.insert(cache_key, collision_resistant_name.clone());
                        collision_resistant_name
                    };
                    processed_image_basenames.insert(sanitized_name);
                }

                // Update the progress counter
                pb.inc(1);

                // Update message periodically to show progress without excessive lock contention
                let count = processed_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                let update_counter =
                    message_update_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if update_counter % MESSAGE_UPDATE_INTERVAL == 0 {
                    pb.set_message(format!("Processed {} files...", count));
                }
            } else if let Some(mut streaming_annotation) =
                read_and_parse_json_streaming(&json_path, args.buffer_size_kib)
            {
                // Determine the image path relative to the JSON file's directory
                let image_path = json_path
                    .parent()
                    .map(|parent| parent.join(&streaming_annotation.image_path))
                    .unwrap_or_else(|| dirname.join(&streaming_annotation.image_path));

                // Add labels to the label map if not using a predefined list and not in deterministic mode
                if args.label_list.is_empty() && !args.deterministic_labels {
                    // Collect unique labels from this annotation to avoid repeated cloning and contention
                    let unique_labels: std::collections::HashSet<&str> = streaming_annotation
                        .shapes
                        .iter()
                        .map(|shape| shape.label.as_str())
                        .collect();

                    // Insert each unique label once per file
                    for label in unique_labels {
                        label_map.entry(label.to_string()).or_insert_with(|| {
                            next_class_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                        });
                    }
                }

                // Process the annotation immediately (copy image, generate label file)
                // Distribute files according to split ratios using a hash-based approach
                let (labels_dir, images_dir) = {
                    // Use a hash of the image path for consistent distribution
                    let hash = {
                        use std::collections::hash_map::DefaultHasher;
                        use std::hash::{Hash, Hasher};
                        let mut hasher = DefaultHasher::new();
                        image_path.hash(&mut hasher);
                        hasher.finish()
                    };
                    let ratio = (hash % 1000) as f32 / 1000.0;

                    if ratio < args.val_size {
                        // Validation set
                        (&output_dirs.val_labels_dir, &output_dirs.val_images_dir)
                    } else if ratio < args.val_size + args.test_size {
                        // Test set
                        (
                            output_dirs
                                .test_labels_dir
                                .as_ref()
                                .unwrap_or(&output_dirs.train_labels_dir),
                            output_dirs
                                .test_images_dir
                                .as_ref()
                                .unwrap_or(&output_dirs.train_images_dir),
                        )
                    } else {
                        // Training set
                        (&output_dirs.train_labels_dir, &output_dirs.train_images_dir)
                    }
                };

                let processor = crate::conversion::AnnotationProcessor {
                    image_path: &image_path,
                    annotation: None,
                    streaming_annotation: Some(&mut streaming_annotation),
                    labels_dir,
                    images_dir,
                    label_map,
                    args,
                    filename_cache: &filename_cache,
                    base_dir: dirname,
                    stats: Some(&mut stats.lock().unwrap()),
                };
                if let Err(e) = crate::conversion::process_annotation(processor) {
                    log::error!(
                        "Failed to process annotation {}: {}",
                        json_path.display(),
                        e
                    );
                }

                // Track processed image basenames for background image detection
                // Use the relative path as the cache key
                let relative_path = image_path
                    .strip_prefix(dirname)
                    .unwrap_or(image_path.as_path());
                let cache_key = relative_path.to_string_lossy().to_string();

                // Use the sanitized name for collision-free tracking
                if let Some(file_stem) = image_path.file_stem().and_then(|s| s.to_str()) {
                    let sanitized_name = if let Some(cached) = filename_cache.get(&cache_key) {
                        cached.clone()
                    } else {
                        // Generate a collision-resistant name using the file stem and relative path
                        let collision_resistant_name =
                            crate::utils::generate_collision_resistant_name(
                                file_stem,
                                relative_path,
                            );
                        filename_cache.insert(cache_key, collision_resistant_name.clone());
                        collision_resistant_name
                    };
                    processed_image_basenames.insert(sanitized_name);
                }

                // Update the progress counter
                pb.inc(1);

                // Update message periodically to show progress without excessive lock contention
                let count = processed_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                let update_counter =
                    message_update_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if update_counter % MESSAGE_UPDATE_INTERVAL == 0 {
                    pb.set_message(format!("Processed {} files...", count));
                }
            } else if let Some(annotation) = read_and_parse_json(&json_path, args.buffer_size_kib) {
                // Fallback to regular parsing if streaming parsing fails
                // Determine the image path relative to the JSON file's directory
                let image_path = json_path
                    .parent()
                    .map(|parent| parent.join(&annotation.image_path))
                    .unwrap_or_else(|| dirname.join(&annotation.image_path));

                // Add labels to the label map if not using a predefined list and not in deterministic mode
                if args.label_list.is_empty() && !args.deterministic_labels {
                    // Collect unique labels from this annotation to avoid repeated cloning and contention
                    let unique_labels: std::collections::HashSet<&str> = annotation
                        .shapes
                        .iter()
                        .map(|shape| shape.label.as_str())
                        .collect();

                    // Insert each unique label once per file
                    for label in unique_labels {
                        label_map.entry(label.to_string()).or_insert_with(|| {
                            next_class_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                        });
                    }
                }

                // Process the annotation immediately (copy image, generate label file)
                // Distribute files according to split ratios using a hash-based approach
                let (labels_dir, images_dir) = {
                    // Use a hash of the image path for consistent distribution
                    let hash = {
                        use std::collections::hash_map::DefaultHasher;
                        use std::hash::{Hash, Hasher};
                        let mut hasher = DefaultHasher::new();
                        image_path.hash(&mut hasher);
                        hasher.finish()
                    };
                    let ratio = (hash % 1000) as f32 / 1000.0;

                    if ratio < args.val_size {
                        // Validation set
                        (&output_dirs.val_labels_dir, &output_dirs.val_images_dir)
                    } else if ratio < args.val_size + args.test_size {
                        // Test set
                        (
                            output_dirs
                                .test_labels_dir
                                .as_ref()
                                .unwrap_or(&output_dirs.train_labels_dir),
                            output_dirs
                                .test_images_dir
                                .as_ref()
                                .unwrap_or(&output_dirs.train_images_dir),
                        )
                    } else {
                        // Training set
                        (&output_dirs.train_labels_dir, &output_dirs.train_images_dir)
                    }
                };

                let processor = crate::conversion::AnnotationProcessor {
                    image_path: &image_path,
                    annotation: Some(&annotation),
                    streaming_annotation: None,
                    labels_dir,
                    images_dir,
                    label_map,
                    args,
                    filename_cache: &filename_cache,
                    base_dir: dirname,
                    stats: Some(&mut stats.lock().unwrap()),
                };
                if let Err(e) = crate::conversion::process_annotation(processor) {
                    log::error!(
                        "Failed to process annotation {}: {}",
                        json_path.display(),
                        e
                    );
                }

                // Track processed image basenames for background image detection
                // Use the relative path as the cache key
                let relative_path = image_path
                    .strip_prefix(dirname)
                    .unwrap_or(image_path.as_path());
                let cache_key = relative_path.to_string_lossy().to_string();

                // Use the sanitized name for collision-free tracking
                if let Some(file_stem) = image_path.file_stem().and_then(|s| s.to_str()) {
                    let sanitized_name = if let Some(cached) = filename_cache.get(&cache_key) {
                        cached.clone()
                    } else {
                        // Generate a collision-resistant name using the file stem and relative path
                        let collision_resistant_name =
                            crate::utils::generate_collision_resistant_name(
                                file_stem,
                                relative_path,
                            );
                        filename_cache.insert(cache_key, collision_resistant_name.clone());
                        collision_resistant_name
                    };
                    processed_image_basenames.insert(sanitized_name);
                }

                // Update the progress counter
                pb.inc(1);

                // Update message periodically to show progress without excessive lock contention
                let count = processed_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                let update_counter =
                    message_update_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if update_counter % MESSAGE_UPDATE_INTERVAL == 0 {
                    pb.set_message(format!("Processed {} files...", count));
                }
            }
        });
    });

    // Finish the progress bar
    pb.finish();

    // Convert the DashSet to a regular HashSet and extract statistics
    let processed_image_basenames = processed_image_basenames.into_iter().collect();
    let final_stats = match Arc::try_unwrap(stats) {
        Ok(mutex) => mutex.into_inner().unwrap(),
        Err(arc) => arc.lock().unwrap().clone(),
    };
    (processed_image_basenames, final_stats)
}

/// Process background images in a second pass
/// This function processes images that don't have corresponding JSON annotations
pub fn process_background_images(
    dirname: &Path,
    args: &Args,
    output_dirs: &OutputDirs,
    label_map: &dashmap::DashMap<String, usize>,
    processed_image_basenames: &std::collections::HashSet<String>,
) {
    // If include_background is not set, we don't need to process background images
    if !args.include_background {
        return;
    }

    // Create a custom thread pool with limited concurrency
    let thread_pool = create_io_thread_pool(args.workers);

    // Use the precomputed set of supported image extensions for fast lookup
    let image_extensions = crate::types::get_image_extensions_set();

    // Walk through the directory structure to find image files using parallel jwalk
    let image_entries = WalkDir::new(dirname)
        .skip_hidden(false)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            // Skip YOLODataset directories early by comparing directory names directly
            if e.file_type().is_dir() {
                if let Some(name) = e.file_name().to_str() {
                    name != "YOLODataset"
                } else {
                    // If we can't convert to str, skip to be safe
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

    // Create a counter for tracking processed background images
    let bg_processed_count = std::sync::atomic::AtomicUsize::new(0);
    // Counter for periodic message updates (every 100 files)
    let bg_message_update_count = std::sync::atomic::AtomicUsize::new(0);
    const BG_MESSAGE_UPDATE_INTERVAL: usize = 100;

    // Create a progress bar for background image processing
    // Since we're using par_bridge(), we can't know the exact count upfront
    // We'll use an indeterminate progress bar instead
    let pb = indicatif::ProgressBar::new_spinner();
    pb.set_style(
        indicatif::ProgressStyle::default_spinner()
            .template("{spinner:.green} [{elapsed_precise}] [Processing background images] {msg}")
            .progress_chars("#>-"),
    );
    pb.enable_steady_tick(100);
    pb.set_message("Processing background images...");

    // Create a cache for sanitized filenames to avoid recomputing
    let filename_cache: Arc<DashMap<String, String>> = Arc::new(DashMap::new());

    // Process background images in parallel using the custom thread pool
    thread_pool.install(|| {
        image_entries.for_each(|image_path| {
            // Check if this image was already processed (has a JSON annotation)
            // Use the relative path as the cache key
            let relative_path = image_path
                .strip_prefix(dirname)
                .unwrap_or(image_path.as_path());
            let cache_key = relative_path.to_string_lossy().to_string();

            // Use the sanitized name for collision-free tracking
            if let Some(file_stem) = image_path.file_stem().and_then(|s| s.to_str()) {
                let sanitized_name = if let Some(cached) = filename_cache.get(&cache_key) {
                    cached.clone()
                } else {
                    // Generate a collision-resistant name using the file stem and relative path
                    let collision_resistant_name =
                        crate::utils::generate_collision_resistant_name(file_stem, relative_path);
                    filename_cache.insert(cache_key, collision_resistant_name.clone());
                    collision_resistant_name
                };

                if !processed_image_basenames.contains(&sanitized_name) {
                    // This is a background image, process it
                    // Distribute files according to split ratios using a hash-based approach
                    let (labels_dir, images_dir) = {
                        // Use a hash of the image path for consistent distribution
                        let hash = {
                            use std::collections::hash_map::DefaultHasher;
                            use std::hash::{Hash, Hasher};
                            let mut hasher = DefaultHasher::new();
                            image_path.hash(&mut hasher);
                            hasher.finish()
                        };
                        let ratio = (hash % 1000) as f32 / 1000.0;

                        if ratio < args.val_size {
                            // Validation set
                            (&output_dirs.val_labels_dir, &output_dirs.val_images_dir)
                        } else if ratio < args.val_size + args.test_size {
                            // Test set
                            (
                                output_dirs
                                    .test_labels_dir
                                    .as_ref()
                                    .unwrap_or(&output_dirs.train_labels_dir),
                                output_dirs
                                    .test_images_dir
                                    .as_ref()
                                    .unwrap_or(&output_dirs.train_images_dir),
                            )
                        } else {
                            // Training set
                            (&output_dirs.train_labels_dir, &output_dirs.train_images_dir)
                        }
                    };

                    let processor = crate::conversion::AnnotationProcessor {
                        image_path: &image_path,
                        annotation: None, // No annotation for background images
                        streaming_annotation: None,
                        labels_dir,
                        images_dir,
                        label_map,
                        args,
                        filename_cache: &filename_cache,
                        base_dir: dirname,
                        stats: None, // No stats tracking for background images
                    };
                    if let Err(e) = crate::conversion::process_annotation(processor) {
                        log::error!(
                            "Failed to process background image {}: {}",
                            image_path.display(),
                            e
                        );
                    }

                    // Update the progress counter
                    pb.inc(1);

                    // Update message periodically to show progress without excessive lock contention
                    let count =
                        bg_processed_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    let update_counter = bg_message_update_count
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                        + 1;
                    if update_counter % BG_MESSAGE_UPDATE_INTERVAL == 0 {
                        pb.set_message(format!("Processed {} background images...", count));
                    }
                }
            }
        });
    });

    // Finish the progress bar
    pb.finish();
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
    let mut sorted_labels: Vec<_> = {
        let mut vec = Vec::with_capacity(label_map.len());
        for entry in label_map.iter() {
            vec.push((entry.key().clone(), *entry.value()));
        }
        vec
    };
    sorted_labels.sort_by_key(|&(_, id)| id);

    for (label, id) in sorted_labels {
        yaml_content.push_str(&format!("    {}: {}\n", id, label));
    }
    dataset_yaml.write_all(yaml_content.as_bytes())
}
