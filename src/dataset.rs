use dashmap::DashMap;
use log::info;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::path::Path;
use std::sync::{
    atomic::{AtomicUsize, Ordering::Relaxed},
    Arc,
};

use crate::config::Args;
use crate::conversion::process_annotations_in_parallel;
use crate::io::create_dataset_yaml;
use crate::types::{ImageAnnotation, OutputDirs, SplitData};
use crate::utils::create_progress_bar;

/// Split the annotations into training, validation, and testing sets
pub fn split_annotations(
    annotations: &mut Vec<(std::path::PathBuf, Option<ImageAnnotation>)>,
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

/// Initialize the label map with labels found in the dataset or specified in label_list
pub fn initialize_label_map(
    split_data: &SplitData,
    label_map: &DashMap<String, usize>,
    next_class_id: &Arc<AtomicUsize>,
    args: &Args,
) {
    // If label_list is specified, use it to initialize label_map
    if !args.label_list.is_empty() {
        for (id, label) in args.label_list.iter().enumerate() {
            label_map.insert(label.clone(), id);
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
                if !label_map.contains_key(&shape.label) {
                    let new_id = next_class_id.fetch_add(1, Relaxed);
                    label_map.insert(shape.label.clone(), new_id);
                }
            });
    }
}

/// Process all annotations for train/val/test splits in parallel
pub fn process_all_annotations(
    split_data: &SplitData,
    output_dirs: &OutputDirs,
    label_map: &DashMap<String, usize>,
    args: &Args,
    dirname: &Path,
) {
    // Process train, validation, and test splits in parallel
    let mut handles = Vec::new();

    // Process train split
    if !split_data.train_annotations.is_empty() {
        let train_pb = create_progress_bar(split_data.train_annotations.len() as u64, "Train");
        let train_annotations = split_data.train_annotations.clone();
        let train_labels_dir = output_dirs.train_labels_dir.clone();
        let train_images_dir = output_dirs.train_images_dir.clone();
        let label_map = label_map.clone();
        let args = args.clone();
        let dirname = dirname.to_path_buf();

        handles.push(std::thread::spawn(move || {
            process_annotations_in_parallel(
                &train_annotations,
                &train_labels_dir,
                &train_images_dir,
                &label_map,
                &args,
                &dirname,
                &train_pb,
            );
            train_pb.finish_with_message("Train processing complete");
        }));
    }

    // Process validation split
    if !split_data.val_annotations.is_empty() {
        let val_pb = create_progress_bar(split_data.val_annotations.len() as u64, "Val");
        let val_annotations = split_data.val_annotations.clone();
        let val_labels_dir = output_dirs.val_labels_dir.clone();
        let val_images_dir = output_dirs.val_images_dir.clone();
        let label_map = label_map.clone();
        let args = args.clone();
        let dirname = dirname.to_path_buf();

        handles.push(std::thread::spawn(move || {
            process_annotations_in_parallel(
                &val_annotations,
                &val_labels_dir,
                &val_images_dir,
                &label_map,
                &args,
                &dirname,
                &val_pb,
            );
            val_pb.finish_with_message("Val processing complete");
        }));
    }

    // Process test split if available
    if let (Some(test_labels_dir), Some(test_images_dir)) =
        (&output_dirs.test_labels_dir, &output_dirs.test_images_dir)
    {
        if !split_data.test_annotations.is_empty() {
            let test_pb = create_progress_bar(split_data.test_annotations.len() as u64, "Test");
            let test_annotations = split_data.test_annotations.clone();
            let test_labels_dir = test_labels_dir.clone();
            let test_images_dir = test_images_dir.clone();
            let label_map = label_map.clone();
            let args = args.clone();
            let dirname = dirname.to_path_buf();

            handles.push(std::thread::spawn(move || {
                process_annotations_in_parallel(
                    &test_annotations,
                    &test_labels_dir,
                    &test_images_dir,
                    &label_map,
                    &args,
                    &dirname,
                    &test_pb,
                );
                test_pb.finish_with_message("Test processing complete");
            }));
        }
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
}

/// Main dataset processing pipeline
pub fn process_dataset(
    mut annotations: Vec<(std::path::PathBuf, Option<ImageAnnotation>)>,
    output_dirs: &OutputDirs,
    args: &Args,
    dirname: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Read and parsed {} files.", annotations.len());

    let split_data = split_annotations(&mut annotations, args.val_size, args.test_size, args.seed);

    let label_map = DashMap::new();
    let next_class_id = Arc::new(AtomicUsize::new(0));

    // Preinitialize the label map with all possible labels from the dataset
    initialize_label_map(&split_data, &label_map, &next_class_id, args);

    process_all_annotations(&split_data, output_dirs, &label_map, args, dirname);

    info!("Creating dataset.yaml file...");
    if let Err(e) = create_dataset_yaml(dirname, args, &label_map) {
        return Err(format!("Failed to create dataset.yaml: {}", e).into());
    } else {
        info!("Conversion process completed successfully.");
    }

    Ok(())
}
