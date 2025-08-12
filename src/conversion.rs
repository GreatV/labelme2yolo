use dashmap::DashMap;
use indicatif::ProgressBar;
use log::{error, warn};
use rayon::prelude::*;
use std::fs::{copy, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::config::{Args, Format};
use crate::types::{ImageAnnotation, Shape};
use crate::utils::infer_image_format;

/// Process a batch of annotations in parallel
pub fn process_annotations_in_parallel(
    annotations: &[(std::path::PathBuf, Option<ImageAnnotation>)],
    labels_dir: &Path,
    images_dir: &Path,
    label_map: &DashMap<String, usize>,
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

/// Process a single annotation and convert it to YOLO format
pub fn process_annotation(
    image_path: &Path,
    annotation: Option<&ImageAnnotation>,
    labels_dir: &Path,
    images_dir: &Path,
    label_map: &DashMap<String, usize>,
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
        copy(image_path, &image_output_path)?;
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

/// Convert an annotation to YOLO format (bounding box or polygon)
pub fn convert_to_yolo_format(
    annotation: &ImageAnnotation,
    args: &Args,
    label_map: &DashMap<String, usize>,
) -> String {
    let mut yolo_data = String::with_capacity(annotation.shapes.len() * 64);

    for shape in &annotation.shapes {
        let class_id = match label_map.get(&shape.label) {
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

/// Process polygon shape data for YOLO format
pub fn process_polygon_shape(yolo_data: &mut String, annotation: &ImageAnnotation, shape: &Shape) {
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

/// Calculate bounding box for YOLO format
pub fn calculate_bounding_box(annotation: &ImageAnnotation, shape: &Shape) -> (f64, f64, f64, f64) {
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
