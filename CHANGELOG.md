# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of the `CHANGELOG.md` file to track changes, improvements, and bug fixes.

### Changed

### Fixed

## [0.2.6] - 2022-12-01

### Added
- Added support for exporting data as YOLO polygon annotation (for YOLOv5 & YOLOv8 segmentation).
- Added option to choose the output format of the label text (`polygon` or `bbox`).

### Changed
- Improved performance by implementing the tool in Rust, making it significantly faster than equivalent Python implementations.

### Fixed

## [0.2.5] - 2022-11-15

### Added
- Initial release of Labelme2YOLO.
