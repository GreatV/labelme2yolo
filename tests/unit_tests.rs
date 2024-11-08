#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File};
    use std::io::Write;
    use std::path::PathBuf;

    #[test]
    fn test_infer_image_format() {
        let jpg_bytes = vec![0xFF, 0xD8, 0xFF];
        let png_bytes = vec![0x89, b'P', b'N', b'G'];
        let bmp_bytes = vec![b'B', b'M'];
        let gif_bytes = vec![0x47, 0x49, 0x46];
        let unknown_bytes = vec![0x00, 0x00, 0x00];

        assert_eq!(infer_image_format(&jpg_bytes), Some("jpg"));
        assert_eq!(infer_image_format(&png_bytes), Some("png"));
        assert_eq!(infer_image_format(&bmp_bytes), Some("bmp"));
        assert_eq!(infer_image_format(&gif_bytes), Some("gif"));
        assert_eq!(infer_image_format(&unknown_bytes), None);
    }

    #[test]
    fn test_validate_size() {
        assert!(validate_size("0.5").is_ok());
        assert!(validate_size("1.0").is_ok());
        assert!(validate_size("0.0").is_ok());
        assert!(validate_size("-0.1").is_err());
        assert!(validate_size("1.1").is_err());
        assert!(validate_size("abc").is_err());
    }

    #[test]
    fn test_split_annotations() {
        let mut annotations = vec![
            (PathBuf::from("image1.jpg"), None),
            (PathBuf::from("image2.jpg"), None),
            (PathBuf::from("image3.jpg"), None),
            (PathBuf::from("image4.jpg"), None),
            (PathBuf::from("image5.jpg"), None),
        ];

        let split_data = split_annotations(&mut annotations, 0.2, 0.2, 42);

        assert_eq!(split_data.train_annotations.len(), 3);
        assert_eq!(split_data.val_annotations.len(), 1);
        assert_eq!(split_data.test_annotations.len(), 1);
    }

    #[test]
    fn test_calculate_bounding_box() {
        let annotation = ImageAnnotation {
            version: "1.0".to_string(),
            flags: None,
            shapes: vec![],
            image_path: "image.jpg".to_string(),
            image_data: None,
            image_height: 100,
            image_width: 100,
        };

        let shape = Shape {
            label: "test".to_string(),
            points: vec![(10.0, 10.0), (20.0, 20.0)],
            group_id: None,
            shape_type: "rectangle".to_string(),
            description: None,
            mask: None,
        };

        let (x_center, y_center, width, height) = calculate_bounding_box(&annotation, &shape);

        assert_eq!(x_center, 0.15);
        assert_eq!(y_center, 0.15);
        assert_eq!(width, 0.1);
        assert_eq!(height, 0.1);
    }

    #[test]
    fn test_convert_to_yolo_format() {
        let annotation = ImageAnnotation {
            version: "1.0".to_string(),
            flags: None,
            shapes: vec![Shape {
                label: "test".to_string(),
                points: vec![(10.0, 10.0), (20.0, 20.0)],
                group_id: None,
                shape_type: "rectangle".to_string(),
                description: None,
                mask: None,
            }],
            image_path: "image.jpg".to_string(),
            image_data: None,
            image_height: 100,
            image_width: 100,
        };

        let args = Args {
            json_dir: "test_dir".to_string(),
            val_size: 0.2,
            test_size: 0.2,
            output_format: Format::Bbox,
            seed: 42,
            include_background: false,
            label_list: vec!["test".to_string()],
        };

        let label_map = Arc::new(Mutex::new(HashMap::new()));
        label_map.lock().unwrap().insert("test".to_string(), 0);

        let yolo_data = convert_to_yolo_format(&annotation, &args, &label_map);

        assert_eq!(yolo_data, "0 0.15 0.15 0.1 0.1\n");
    }

    #[test]
    fn test_create_dataset_yaml() {
        let temp_dir = tempfile::tempdir().unwrap();
        let dirname = temp_dir.path().to_path_buf();

        let args = Args {
            json_dir: "test_dir".to_string(),
            val_size: 0.2,
            test_size: 0.2,
            output_format: Format::Bbox,
            seed: 42,
            include_background: false,
            label_list: vec!["test".to_string()],
        };

        let label_map = Arc::new(Mutex::new(HashMap::new()));
        label_map.lock().unwrap().insert("test".to_string(), 0);

        create_dataset_yaml(&dirname, &args, &label_map).unwrap();

        let dataset_yaml_path = dirname.join("YOLODataset/dataset.yaml");
        let yaml_content = fs::read_to_string(dataset_yaml_path).unwrap();

        assert!(yaml_content.contains("path:"));
        assert!(yaml_content.contains("train: images/train"));
        assert!(yaml_content.contains("val: images/val"));
        assert!(yaml_content.contains("test: images/test"));
        assert!(yaml_content.contains("names:"));
        assert!(yaml_content.contains("0: test"));
    }
}
