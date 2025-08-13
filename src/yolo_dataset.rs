use dashmap::{DashMap, DashSet};
use log::info;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{
    atomic::{AtomicUsize, Ordering::Relaxed},
    Arc,
};

use crate::config::Args;
use crate::io::{create_dataset_yaml, process_background_images, process_json_files_streaming};
use crate::types::{OutputDirs, SplitData};

/// Main dataset processing pipeline using streaming approach to reduce memory footprint
pub fn process_dataset(
    output_dirs: &OutputDirs,
    args: &Args,
    dirname: &Path,
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

    // Create a dummy SplitData for compatibility with existing functions
    // In the streaming approach, we don't split the data upfront
    let mut split_data = SplitData {
        train_annotations: Vec::new(),
        val_annotations: Vec::new(),
        test_annotations: Vec::new(),
    };

    let (processed_image_basenames, stats): (
        std::collections::HashSet<String>,
        crate::types::ProcessingStats,
    );

    if args.deterministic_labels && args.label_list.is_empty() {
        // Two-pass mode for deterministic label ID assignment
        info!("Starting two-pass processing for deterministic label mapping...");

        // Pass 1: Gather label vocabulary only
        info!("Pass 1: Gathering label vocabulary...");
        let all_labels = gather_label_vocabulary(dirname, args)?;
        info!("Found {} unique labels.", all_labels.len());

        // Sort labels alphabetically for deterministic ID assignment
        let mut sorted_labels: Vec<String> = all_labels.into_iter().collect();
        sorted_labels.sort();

        // Assign IDs deterministically
        for (id, label) in sorted_labels.into_iter().enumerate() {
            label_map.insert(label, id);
        }
        next_class_id.store(label_map.len(), Relaxed);

        // Pass 2: Process files with pre-populated label map
        info!("Pass 2: Processing files with deterministic label mapping...");
        (processed_image_basenames, stats) = process_json_files_streaming(
            dirname,
            args,
            output_dirs,
            &label_map,
            &next_class_id,
            &mut split_data,
        );
        info!("Processed {} JSON files.", processed_image_basenames.len());
        stats.print_summary();
    } else {
        // Single pass mode (existing behavior)
        info!("Processing JSON files in streaming fashion...");
        (processed_image_basenames, stats) = process_json_files_streaming(
            dirname,
            args,
            output_dirs,
            &label_map,
            &next_class_id,
            &mut split_data,
        );
        info!("Processed {} JSON files.", processed_image_basenames.len());
        stats.print_summary();
    }

    // Second pass: process background images
    info!("Processing background images...");
    process_background_images(
        dirname,
        args,
        output_dirs,
        &label_map,
        &processed_image_basenames, // Pass the actual processed image basenames from the first pass
    );
    info!("Background image processing complete.");

    info!("Creating dataset.yaml file...");
    if let Err(e) = create_dataset_yaml(dirname, args, &label_map) {
        return Err(format!("Failed to create dataset.yaml: {}", e).into());
    } else {
        info!("Conversion process completed successfully.");
    }

    Ok(())
}

/// Gather all unique labels from JSON files without processing them
fn gather_label_vocabulary(
    dirname: &Path,
    _args: &Args,
) -> Result<std::collections::HashSet<String>, Box<dyn std::error::Error>> {
    use jwalk::WalkDir;
    use rayon::prelude::*;
    use std::collections::HashSet;

    let all_labels: DashSet<String> = DashSet::new();

    // Walk through the directory structure to find JSON files
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

    // Process JSON files in parallel to gather labels
    json_entries.for_each(|json_path| {
        // Try to parse with simd-json for faster performance
        if let Some(annotation) = crate::utils::read_and_parse_json_buffered(&json_path, 64) {
            // Collect unique labels from this annotation
            let labels: HashSet<&str> = annotation
                .shapes
                .iter()
                .map(|shape| shape.label.as_str())
                .collect();

            // Add to the global set
            for label in labels {
                all_labels.insert(label.to_string());
            }
        } else if let Some(streaming_annotation) =
            crate::utils::read_and_parse_json_streaming(&json_path, 64)
        {
            // Collect unique labels from this streaming annotation
            let labels: HashSet<&str> = streaming_annotation
                .shapes
                .iter()
                .map(|shape| shape.label.as_str())
                .collect();

            // Add to the global set
            for label in labels {
                all_labels.insert(label.to_string());
            }
        } else if let Some(annotation) = crate::utils::read_and_parse_json(&json_path, 64) {
            // Fallback to regular parsing
            // Collect unique labels from this annotation
            let labels: HashSet<&str> = annotation
                .shapes
                .iter()
                .map(|shape| shape.label.as_str())
                .collect();

            // Add to the global set
            for label in labels {
                all_labels.insert(label.to_string());
            }
        }
    });

    Ok(all_labels.into_iter().collect())
}
