use clap::Parser;

use log::{error, info};

use labelme2yolo::utils::resolve_source_dirs;
use labelme2yolo::{process_dataset, setup_output_directories, Args, SourceRoot};

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

    info!("Starting the conversion process...");

    match setup_output_directories(&args, &source_roots[0].path) {
        Ok(output_dirs) => {
            if let Err(e) = process_dataset(&output_dirs, &args, &source_roots) {
                error!("Failed to process dataset: {}", e);
            }
        }
        Err(e) => error!("Failed to set up output directories: {}", e),
    }
}
