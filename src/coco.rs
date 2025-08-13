//! COCO format data structures and utilities
//!
//! This module provides functionality to convert LabelMe annotations to COCO format
//! for instance segmentation and object detection.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// COCO dataset information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Info {
    pub year: u32,
    pub version: String,
    pub description: String,
    pub contributor: String,
    pub url: String,
    pub date_created: String,
}

impl Default for Info {
    fn default() -> Self {
        Self {
            year: 2024,
            version: "1.0".to_string(),
            description: "Exported from LabelMe".to_string(),
            contributor: "labelme2coco".to_string(),
            url: String::new(),
            date_created: chrono::Utc::now().date_naive().to_string(),
        }
    }
}

/// COCO license information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct License {
    pub id: u32,
    pub name: String,
    pub url: String,
}

impl Default for License {
    fn default() -> Self {
        Self {
            id: 1,
            name: "Unknown".to_string(),
            url: String::new(),
        }
    }
}

/// COCO category information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Category {
    pub id: u32,
    pub name: String,
    pub supercategory: String,
}

/// COCO image information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Image {
    pub id: u32,
    pub file_name: String,
    pub width: u32,
    pub height: u32,
    pub license: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flickr_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub coco_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub date_captured: Option<String>,
}

impl Image {
    pub fn new(id: u32, file_name: String, width: u32, height: u32) -> Self {
        Self {
            id,
            file_name,
            width,
            height,
            license: 1,
            flickr_url: None,
            coco_url: None,
            date_captured: None,
        }
    }
}

/// COCO annotation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Annotation {
    pub id: u32,
    pub image_id: u32,
    pub category_id: u32,
    pub bbox: [f64; 4], // [x, y, width, height]
    pub area: f64,
    pub iscrowd: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segmentation: Option<Vec<Vec<f64>>>,
}

impl Annotation {
    pub fn new(
        id: u32,
        image_id: u32,
        category_id: u32,
        bbox: [f64; 4],
        area: f64,
        iscrowd: u32,
        segmentation: Option<Vec<Vec<f64>>>,
    ) -> Self {
        Self {
            id,
            image_id,
            category_id,
            bbox,
            area,
            iscrowd,
            segmentation,
        }
    }
}

/// Complete COCO dataset structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CocoFile {
    pub info: Info,
    pub licenses: Vec<License>,
    pub categories: Vec<Category>,
    pub images: Vec<Image>,
    pub annotations: Vec<Annotation>,
}

impl Default for CocoFile {
    fn default() -> Self {
        Self {
            info: Info::default(),
            licenses: vec![License::default()],
            categories: Vec::new(),
            images: Vec::new(),
            annotations: Vec::new(),
        }
    }
}

// Re-export from config module to avoid duplication
pub use crate::config::{CocoConfig, SegmentationMode};

/// Writer for COCO format datasets
pub struct CocoWriter {
    _config: CocoConfig,
    next_image_id: u32,
    next_annotation_id: u32,
    categories: Vec<Category>,
    image_map: HashMap<String, u32>, // Maps file_name to image_id
}

impl CocoWriter {
    /// Create a new COCO writer with the given configuration
    pub fn new(config: CocoConfig) -> Self {
        let start_image_id = config.start_image_id;
        let start_annotation_id = config.start_annotation_id;
        Self {
            _config: config,
            next_image_id: start_image_id,
            next_annotation_id: start_annotation_id,
            categories: Vec::new(),
            image_map: HashMap::new(),
        }
    }

    /// Add categories to the COCO dataset
    pub fn add_categories(&mut self, label_map: &HashMap<String, usize>) {
        self.categories.clear();
        for (label, &id) in label_map {
            self.categories.push(Category {
                id: (id + 1) as u32, // COCO uses 1-based indexing
                name: label.clone(),
                supercategory: "none".to_string(),
            });
        }
        // Sort by ID to ensure consistent ordering
        self.categories.sort_by_key(|c| c.id);
    }

    /// Add an image to the COCO dataset
    pub fn add_image(&mut self, file_name: String, _width: u32, _height: u32) -> u32 {
        let image_id = self.next_image_id;
        self.next_image_id += 1;
        self.image_map.insert(file_name.clone(), image_id);
        image_id
    }

    /// Add an annotation to the COCO dataset
    pub fn add_annotation(
        &mut self,
        _image_id: u32,
        _category_id: u32,
        _bbox: [f64; 4],
        _area: f64,
        _iscrowd: u32,
        _segmentation: Option<Vec<Vec<f64>>>,
    ) -> u32 {
        let annotation_id = self.next_annotation_id;
        self.next_annotation_id += 1;
        annotation_id
    }

    /// Get the image ID for a given file name
    pub fn get_image_id(&self, file_name: &str) -> Option<u32> {
        self.image_map.get(file_name).copied()
    }

    /// Build the complete COCO dataset structure
    pub fn build(&self, images: Vec<Image>, annotations: Vec<Annotation>) -> CocoFile {
        CocoFile {
            info: Info::default(),
            licenses: vec![License::default()],
            categories: self.categories.clone(),
            images,
            annotations,
        }
    }
}

/// Calculate polygon area using the shoelace formula
pub fn calculate_polygon_area(polygon: &[f64]) -> f64 {
    if polygon.len() < 6 || polygon.len() % 2 != 0 {
        return 0.0;
    }

    let mut area = 0.0;
    let n = polygon.len() / 2;

    for i in 0..n {
        let j = (i + 1) % n;
        let x_i = polygon[i * 2];
        let y_i = polygon[i * 2 + 1];
        let x_j = polygon[j * 2];
        let y_j = polygon[j * 2 + 1];
        area += x_i * y_j - x_j * y_i;
    }

    area.abs() / 2.0
}

/// Calculate bounding box from polygon points
pub fn calculate_bbox_from_polygon(polygon: &[f64]) -> [f64; 4] {
    if polygon.len() < 6 || polygon.len() % 2 != 0 {
        return [0.0, 0.0, 0.0, 0.0];
    }

    let mut min_x = f64::MAX;
    let mut min_y = f64::MAX;
    let mut max_x = f64::MIN;
    let mut max_y = f64::MIN;

    for i in 0..polygon.len() / 2 {
        let x = polygon[i * 2];
        let y = polygon[i * 2 + 1];
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }

    [min_x, min_y, max_x - min_x, max_y - min_y]
}

/// Convert a rectangle to polygon points
pub fn rectangle_to_polygon(x1: f64, y1: f64, x2: f64, y2: f64) -> Vec<f64> {
    vec![x1, y1, x2, y1, x2, y2, x1, y2]
}

/// Convert a circle to polygon points
pub fn circle_to_polygon(cx: f64, cy: f64, radius: f64, num_points: usize) -> Vec<f64> {
    let mut points = Vec::with_capacity(num_points * 2);

    for i in 0..num_points {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / num_points as f64;
        let x = cx + radius * angle.cos();
        let y = cy + radius * angle.sin();
        points.push(x);
        points.push(y);
    }

    points
}

/// Clamp coordinates to image bounds
pub fn clamp_coords(x: f64, y: f64, width: u32, height: u32) -> (f64, f64) {
    let x = x.max(0.0).min(width as f64);
    let y = y.max(0.0).min(height as f64);
    (x, y)
}
