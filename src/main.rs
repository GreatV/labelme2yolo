use clap::Parser;

use log::{error, info};
use std::path::PathBuf;

use labelme2yolo::{process_dataset, read_and_parse_json_files, setup_output_directories, Args};

fn main() {
    // Initialize the logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    let dirname = PathBuf::from(&args.json_dir);
    if !dirname.exists() {
        error!("The specified json_dir does not exist: {}", args.json_dir);
        return;
    }

    info!("Starting the conversion process...");

    match setup_output_directories(&args, &dirname) {
        Ok(output_dirs) => {
            let annotations = read_and_parse_json_files(&dirname, &args);

            if let Err(e) = process_dataset(annotations, &output_dirs, &args, &dirname) {
                error!("Failed to process dataset: {}", e);
            }
        }
        Err(e) => error!("Failed to set up output directories: {}", e),
    }
}
