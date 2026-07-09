use clap::Parser;
use log::{error, info};

use labelme2yolo::utils::resolve_source_dirs;
use labelme2yolo::{config::Args, process_coco_dataset, setup_coco_output_directories, SourceRoot};

fn main() {
    // Initialize the logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    let dirs = match resolve_source_dirs(&args) {
        Ok(dirs) => dirs,
        Err(e) => {
            error!("{}", e);
            return;
        }
    };
    let source_roots = SourceRoot::from_dirs(&dirs);

    info!("Starting LabelMe to COCO conversion process...");

    // Parse COCO-specific configuration
    let coco_config = match args.to_coco_config() {
        Ok(config) => config,
        Err(e) => {
            error!("Failed to parse COCO configuration: {}", e);
            return;
        }
    };

    match setup_coco_output_directories(&args, &source_roots[0].path) {
        Ok(output_dirs) => {
            if let Err(e) = process_coco_dataset(&output_dirs, &args, &source_roots, &coco_config) {
                error!("Failed to process dataset: {}", e);
            } else {
                info!("COCO conversion process completed successfully.");
            }
        }
        Err(e) => error!("Failed to set up output directories: {}", e),
    }
}
