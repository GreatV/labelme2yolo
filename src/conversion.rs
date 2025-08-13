use dashmap::DashMap;
use log::{debug, warn};
use std::fs::{copy, File};
use std::io::{BufWriter, Cursor, Write};
use std::path::Path;
use std::sync::Arc;

use crate::config::{Args, Format};
use crate::types::{ImageAnnotation, Shape, StreamingImageAnnotation};
use crate::utils::{generate_collision_resistant_name, infer_image_format};

/// Configuration for processing a single annotation
pub struct AnnotationProcessor<'a> {
    pub image_path: &'a Path,
    pub annotation: Option<&'a ImageAnnotation>,
    pub streaming_annotation: Option<&'a mut StreamingImageAnnotation>,
    pub labels_dir: &'a Path,
    pub images_dir: &'a Path,
    pub label_map: &'a dashmap::DashMap<String, usize>,
    pub args: &'a Args,
    pub filename_cache: &'a Arc<DashMap<String, String>>,
    pub base_dir: &'a Path,
    pub stats: Option<&'a mut crate::types::ProcessingStats>,
}

/// Process a single annotation and convert it to YOLO format
pub fn process_annotation(config: AnnotationProcessor<'_>) -> std::io::Result<()> {
    let AnnotationProcessor {
        image_path,
        annotation,
        streaming_annotation,
        labels_dir,
        images_dir,
        label_map,
        args,
        filename_cache,
        base_dir,
        stats,
    } = config;

    // Calculate the relative path from the base directory
    let relative_path = image_path.strip_prefix(base_dir).unwrap_or(image_path);

    // Use the relative path as the cache key
    let cache_key = relative_path.to_string_lossy().to_string();

    let file_stem = image_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Invalid file stem for path: {:?}", image_path),
            )
        })?;

    let sanitized_name = if let Some(cached) = filename_cache.get(&cache_key) {
        cached.clone()
    } else {
        // Generate a collision-resistant name using the file stem and relative path
        let collision_resistant_name = generate_collision_resistant_name(file_stem, relative_path);
        filename_cache.insert(cache_key, collision_resistant_name.clone());
        collision_resistant_name
    };

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
                // Infer image format by decoding just the header
                let image_extension = {
                    // Create a cursor to read from the base64 string
                    let mut cursor = Cursor::new(image_data);

                    // Create a base64 decoder that reads from the cursor
                    let mut decoder =
                        base64::read::DecoderReader::new(&mut cursor, base64::STANDARD);

                    // Read first 16 bytes to determine image format
                    let mut header_buffer = [0u8; 16];
                    let bytes_read = std::io::Read::read(&mut decoder, &mut header_buffer)
                        .map_err(std::io::Error::other)?;

                    // Infer image format from the header
                    infer_image_format(&header_buffer[..bytes_read]).unwrap_or("png")
                };

                // Create the output file with the correct extension
                let image_output_path = images_dir
                    .join(&sanitized_name)
                    .with_extension(image_extension);

                // Decode the full data and stream it to the file
                let mut cursor = Cursor::new(image_data);
                let mut decoder = base64::read::DecoderReader::new(&mut cursor, base64::STANDARD);

                let mut file = File::create(&image_output_path)?;

                // Copy the decoded data directly to the file (streaming)
                std::io::copy(&mut decoder, &mut file).map_err(std::io::Error::other)?;
            } else {
                // No image data available
                warn!("No image data found in JSON for: {:?}", image_path);
                if let Some(stats) = stats {
                    stats.increment_skipped_no_image_data();
                }
                return Ok(());
            }
        } else {
            // No image data available
            warn!("No image data found in JSON for: {:?}", image_path);
            if let Some(stats) = stats {
                stats.increment_skipped_no_image_data();
            }
            return Ok(());
        }
    } else if let Some(streaming_annotation) = streaming_annotation {
        // Handle missing image file by extracting image_data from streaming JSON
        // Convert streaming annotation to regular annotation first
        let regular_annotation = match streaming_annotation.to_image_annotation() {
            Ok(annotation) => annotation,
            Err(e) => {
                warn!(
                    "Failed to convert streaming annotation for {:?}: {}",
                    image_path, e
                );
                if let Some(stats) = stats {
                    stats.increment_failed();
                }
                return Ok(());
            }
        };

        // Now process the regular annotation
        if let Some(image_data) = &regular_annotation.image_data {
            if !image_data.is_empty() {
                // Infer image format by decoding just the header
                let image_extension = {
                    // Create a cursor to read from the base64 string
                    let mut cursor = Cursor::new(image_data);

                    // Create a base64 decoder that reads from the cursor
                    let mut decoder =
                        base64::read::DecoderReader::new(&mut cursor, base64::STANDARD);

                    // Read first 16 bytes to determine image format
                    let mut header_buffer = [0u8; 16];
                    let bytes_read = std::io::Read::read(&mut decoder, &mut header_buffer)
                        .map_err(std::io::Error::other)?;

                    // Infer image format from the header
                    infer_image_format(&header_buffer[..bytes_read]).unwrap_or("png")
                };

                // Create the output file with the correct extension
                let image_output_path = images_dir
                    .join(&sanitized_name)
                    .with_extension(image_extension);

                // Decode the full data and stream it to the file
                let mut cursor = Cursor::new(image_data);
                let mut decoder = base64::read::DecoderReader::new(&mut cursor, base64::STANDARD);

                let mut file = File::create(&image_output_path)?;

                // Copy the decoded data directly to the file (streaming)
                std::io::copy(&mut decoder, &mut file).map_err(std::io::Error::other)?;
            } else {
                // No image data available
                warn!("No image data found in JSON for: {:?}", image_path);
                if let Some(stats) = stats {
                    stats.increment_skipped_no_image_data();
                }
                return Ok(());
            }
        } else {
            // No image data available
            warn!("No image data found in JSON for: {:?}", image_path);
            if let Some(stats) = stats {
                stats.increment_skipped_no_image_data();
            }
            return Ok(());
        }
    } else {
        // No image file or annotation available
        warn!(
            "Image file not found and no annotation provided for: {:?}",
            image_path
        );
        if let Some(stats) = stats {
            stats.increment_skipped_missing_image();
        }
        return Ok(());
    }

    // Generate label file
    let label_output_path = labels_dir.join(&sanitized_name).with_extension("txt");
    debug!("Creating label file: {:?}", label_output_path);
    let mut writer = BufWriter::new(File::create(&label_output_path)?);

    if let Some(annotation) = annotation {
        // Convert annotation to YOLO format
        convert_to_yolo_format(annotation, args, label_map, &mut writer)?;
    } else if args.include_background {
        // Background image, generate empty label file
        debug!(
            "Creating empty label file for background image: {:?}",
            label_output_path
        );
        writer.write_all(b"")?;
    }

    // Increment successful conversion counter
    if let Some(stats) = stats {
        stats.increment_successful();
    }

    Ok(())
}

/// Convert an annotation to YOLO format (bounding box or polygon)
pub fn convert_to_yolo_format(
    annotation: &ImageAnnotation,
    args: &Args,
    label_map: &dashmap::DashMap<String, usize>,
    writer: &mut BufWriter<File>,
) -> std::io::Result<()> {
    // Precompute normalization factors to avoid repeated divisions
    let inv_w = 1.0 / annotation.image_width as f64;
    let inv_h = 1.0 / annotation.image_height as f64;

    // Pre-format all shapes into strings to minimize write calls
    let mut formatted_shapes = Vec::with_capacity(annotation.shapes.len());

    for shape in &annotation.shapes {
        let class_id = match label_map.get(&shape.label) {
            Some(class_id) => *class_id,
            None => {
                if log::log_enabled!(log::Level::Debug) {
                    debug!("Skipping shape with unknown label: {}", shape.label);
                }
                continue;
            }
        };

        if log::log_enabled!(log::Level::Debug) {
            debug!(
                "Processing shape: {} with class_id: {}",
                shape.label, class_id
            );
        }
        match args.output_format {
            Format::Polygon => {
                let mut shape_str = String::with_capacity(256); // Pre-allocate reasonable capacity
                shape_str.push_str(&class_id.to_string());
                format_polygon_shape(&mut shape_str, annotation, shape, inv_w, inv_h)?;
                shape_str.push('\n');
                formatted_shapes.push(shape_str);
            }
            Format::Bbox => {
                let (x_center, y_center, width, height) =
                    calculate_bounding_box(annotation, shape, inv_w, inv_h);
                if log::log_enabled!(log::Level::Debug) {
                    debug!(
                        "BBox: x={:.6}, y={:.6}, w={:.6}, h={:.6}",
                        x_center, y_center, width, height
                    );
                }
                let mut shape_str = String::with_capacity(64); // Pre-allocate reasonable capacity for bbox
                shape_str.push_str(&class_id.to_string());
                format_bbox_shape(&mut shape_str, x_center, y_center, width, height);
                shape_str.push('\n');
                formatted_shapes.push(shape_str);
            }
        }
    }

    // Write all formatted shapes at once
    for shape_str in formatted_shapes {
        writer.write_all(shape_str.as_bytes())?;
    }

    Ok(())
}

/// Process polygon shape data for YOLO format
pub fn process_polygon_shape(
    writer: &mut BufWriter<File>,
    _annotation: &ImageAnnotation,
    shape: &Shape,
    inv_w: f64,
    inv_h: f64,
) -> std::io::Result<()> {
    if shape.shape_type == "rectangle" {
        let (x1, y1) = shape.points[0];
        let (x2, y2) = shape.points[1];
        let rect_points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)];
        for &(x, y) in &rect_points {
            let x_norm = x * inv_w;
            let y_norm = y * inv_h;
            write!(writer, " {:.6} {:.6}", x_norm, y_norm)?;
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
            let x_norm = x * inv_w;
            let y_norm = y * inv_h;
            write!(writer, " {:.6} {:.6}", x_norm, y_norm)?;
        }
    } else {
        for &(x, y) in &shape.points {
            let x_norm = x * inv_w;
            let y_norm = y * inv_h;
            write!(writer, " {:.6} {:.6}", x_norm, y_norm)?;
        }
    }

    Ok(())
}

/// Format polygon shape data into a string for YOLO format
pub fn format_polygon_shape(
    shape_str: &mut String,
    _annotation: &ImageAnnotation,
    shape: &Shape,
    inv_w: f64,
    inv_h: f64,
) -> std::io::Result<()> {
    if shape.shape_type == "rectangle" {
        let (x1, y1) = shape.points[0];
        let (x2, y2) = shape.points[1];
        let rect_points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)];
        for &(x, y) in &rect_points {
            let x_norm = x * inv_w;
            let y_norm = y * inv_h;
            shape_str.push_str(&format!(" {:.6} {:.6}", x_norm, y_norm));
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
            let x_norm = x * inv_w;
            let y_norm = y * inv_h;
            shape_str.push_str(&format!(" {:.6} {:.6}", x_norm, y_norm));
        }
    } else {
        for &(x, y) in &shape.points {
            let x_norm = x * inv_w;
            let y_norm = y * inv_h;
            shape_str.push_str(&format!(" {:.6} {:.6}", x_norm, y_norm));
        }
    }

    Ok(())
}

/// Format bbox shape data into a string for YOLO format
pub fn format_bbox_shape(
    shape_str: &mut String,
    x_center: f64,
    y_center: f64,
    width: f64,
    height: f64,
) {
    shape_str.push_str(&format!(
        " {:.6} {:.6} {:.6} {:.6}",
        x_center, y_center, width, height
    ));
}

/// Format bbox shape data into a string for YOLO format using ryu for faster float formatting
pub fn format_bbox_shape_ryu(
    shape_str: &mut String,
    x_center: f64,
    y_center: f64,
    width: f64,
    height: f64,
) {
    // Helper function to format a single float value to 6 decimal places
    let format_value = |value: f64, s: &mut String| {
        s.push(' ');
        let formatted = format!("{:.6}", value);
        s.push_str(&formatted);
    };

    format_value(x_center, shape_str);
    format_value(y_center, shape_str);
    format_value(width, shape_str);
    format_value(height, shape_str);
}

/// Calculate bounding box for YOLO format
pub fn calculate_bounding_box(
    _annotation: &ImageAnnotation,
    shape: &Shape,
    inv_w: f64,
    inv_h: f64,
) -> (f64, f64, f64, f64) {
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

    let x_center = (x_min + x_max) * 0.5 * inv_w;
    let y_center = (y_min + y_max) * 0.5 * inv_h;
    let width = (x_max - x_min) * inv_w;
    let height = (y_max - y_min) * inv_h;

    (x_center, y_center, width, height)
}
