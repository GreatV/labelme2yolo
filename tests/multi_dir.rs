//! Integration tests for converting multiple source directories in one run
//! (https://github.com/GreatV/labelme2yolo/issues/87)

use clap::Parser;
use std::fs;
use std::path::Path;

use labelme2yolo::config::{Format, SegmentationMode};
use labelme2yolo::utils::resolve_source_dirs;
use labelme2yolo::{process_dataset, setup_output_directories, Args, SourceRoot};

fn make_args(json_dirs: &[&str], output_dir: Option<&str>) -> Args {
    Args {
        json_dir: json_dirs.iter().map(|s| s.to_string()).collect(),
        output_dir: output_dir.map(|s| s.to_string()),
        val_size: 0.0,
        test_size: 0.0,
        output_format: Format::Bbox,
        seed: 42,
        include_background: false,
        label_list: Vec::new(),
        workers: 1,
        deterministic_labels: false,
        buffer_size_kib: 64,
        segmentation_mode: SegmentationMode::Polygon,
        start_ids: "image=1,ann=1".to_string(),
        categories_from: "inferred".to_string(),
    }
}

/// Write a minimal LabelMe annotation plus a dummy image file into `dir`
fn write_sample(dir: &Path, stem: &str, label: &str) {
    let image_name = format!("{}.png", stem);
    let json = format!(
        r#"{{
            "version": "5.0.1",
            "flags": {{}},
            "shapes": [
                {{
                    "label": "{}",
                    "points": [[10.0, 10.0], [50.0, 50.0]],
                    "group_id": null,
                    "shape_type": "rectangle",
                    "description": null,
                    "mask": null
                }}
            ],
            "imagePath": "{}",
            "imageData": null,
            "imageHeight": 100,
            "imageWidth": 100
        }}"#,
        label, image_name
    );
    fs::write(dir.join(format!("{}.json", stem)), json).unwrap();
    // Content is only copied, never decoded, so any bytes work as an image
    fs::write(dir.join(image_name), b"fake png bytes").unwrap();
}

fn count_files(dir: &Path) -> usize {
    fs::read_dir(dir).map(|d| d.count()).unwrap_or(0)
}

#[test]
fn cli_accepts_repeated_json_dir_flag() {
    let args = Args::try_parse_from(["labelme2yolo", "-d", "src-a", "-d", "src-b", "-o", "out"])
        .expect("repeated -d should parse");
    assert_eq!(args.json_dir, vec!["src-a", "src-b"]);

    assert!(
        Args::try_parse_from(["labelme2yolo"]).is_err(),
        "-d should be required"
    );
}

#[test]
fn multiple_dirs_require_output_dir() {
    let tmp = tempfile::tempdir().unwrap();
    let src_a = tmp.path().join("src-a");
    let src_b = tmp.path().join("src-b");
    fs::create_dir_all(&src_a).unwrap();
    fs::create_dir_all(&src_b).unwrap();

    let args = make_args(&[src_a.to_str().unwrap(), src_b.to_str().unwrap()], None);
    let err = resolve_source_dirs(&args).unwrap_err();
    assert!(err.contains("--output_dir"), "unexpected error: {}", err);
}

#[test]
fn missing_dir_is_rejected_and_duplicates_are_removed() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    fs::create_dir_all(&src).unwrap();
    let src_str = src.to_str().unwrap();

    let args = make_args(&["does-not-exist"], None);
    assert!(resolve_source_dirs(&args).is_err());

    // Programmatic use with no directories at all must error, not panic later
    let args = make_args(&[], None);
    assert!(resolve_source_dirs(&args).is_err());

    let args = make_args(&[src_str, src_str], Some("out"));
    let dirs = resolve_source_dirs(&args).unwrap();
    assert_eq!(dirs.len(), 1);
}

#[test]
fn converts_multiple_dirs_with_identical_file_names() {
    let tmp = tempfile::tempdir().unwrap();
    let src_a = tmp.path().join("src-a");
    let src_b = tmp.path().join("src-b");
    let out = tmp.path().join("out");
    fs::create_dir_all(&src_a).unwrap();
    fs::create_dir_all(&src_b).unwrap();

    // Same file names in both directories: the outputs must not overwrite
    // each other
    write_sample(&src_a, "img", "cat");
    write_sample(&src_b, "img", "dog");

    let args = make_args(
        &[src_a.to_str().unwrap(), src_b.to_str().unwrap()],
        Some(out.to_str().unwrap()),
    );
    let dirs = resolve_source_dirs(&args).unwrap();
    let source_roots = SourceRoot::from_dirs(&dirs);

    let output_dirs = setup_output_directories(&args, &source_roots[0].path).unwrap();
    process_dataset(&output_dirs, &args, &source_roots).unwrap();

    // val_size = 0.0, so everything lands in train
    assert_eq!(count_files(&out.join("images").join("train")), 2);
    assert_eq!(count_files(&out.join("labels").join("train")), 2);

    // Both labels must appear in dataset.yaml
    let yaml = fs::read_to_string(out.join("dataset.yaml")).unwrap();
    assert!(
        yaml.contains("cat"),
        "dataset.yaml missing 'cat':\n{}",
        yaml
    );
    assert!(
        yaml.contains("dog"),
        "dataset.yaml missing 'dog':\n{}",
        yaml
    );
}

#[test]
fn single_dir_conversion_still_works() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    fs::create_dir_all(&src).unwrap();
    write_sample(&src, "img", "cat");

    // No --output_dir: default <json_dir>/YOLODataset is used
    let args = make_args(&[src.to_str().unwrap()], None);
    let dirs = resolve_source_dirs(&args).unwrap();
    let source_roots = SourceRoot::from_dirs(&dirs);

    let output_dirs = setup_output_directories(&args, &source_roots[0].path).unwrap();
    process_dataset(&output_dirs, &args, &source_roots).unwrap();

    let out = src.join("YOLODataset");
    assert_eq!(count_files(&out.join("images").join("train")), 1);
    assert_eq!(count_files(&out.join("labels").join("train")), 1);
}
