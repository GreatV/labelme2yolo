use clap::Parser;
use log::{error, info};
use std::path::PathBuf;

use labelme2yolo::{config::Args, process_coco_dataset, setup_coco_output_directories};

fn main() {
    // Initialize the logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    let dirname = PathBuf::from(&args.json_dir);
    if !dirname.exists() {
        error!("The specified json_dir does not exist: {}", args.json_dir);
        return;
    }

    info!("Starting LabelMe to COCO conversion process...");

    // Parse COCO-specific configuration
    let coco_config = match args.to_coco_config() {
        Ok(config) => config,
        Err(e) => {
            error!("Failed to parse COCO configuration: {}", e);
            return;
        }
    };

    match setup_coco_output_directories(&args, &dirname) {
        Ok(output_dirs) => {
            if let Err(e) = process_coco_dataset(&output_dirs, &args, &dirname, &coco_config) {
                error!("Failed to process dataset: {}", e);
            } else {
                info!("COCO conversion process completed successfully.");
            }
        }
        Err(e) => error!("Failed to set up output directories: {}", e),
    }
}
