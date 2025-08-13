use indicatif::{ProgressBar, ProgressStyle};
use log::error;
use num_cpus;
use rayon;
use serde_json;
use std::fs;
use std::io::{BufReader, Read};
use std::path::Path;
use std::sync::{Arc, Condvar, Mutex};

use crate::types::{ImageAnnotation, StreamingImageAnnotation};

/// Helper function to infer image format from image bytes
pub fn infer_image_format(image_bytes: &[u8]) -> Option<&'static str> {
    if image_bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some("jpg")
    } else if image_bytes.starts_with(&[0x89, b'P', b'N', b'G']) {
        Some("png")
    } else if image_bytes.starts_with(b"BM") {
        Some("bmp")
    } else if image_bytes.starts_with(&[0x47, 0x49, 0x46]) {
        Some("gif")
    } else {
        None
    }
}

/// Read image dimensions from image file header (fast, doesn't load entire image)
pub fn read_image_dimensions(image_path: &Path) -> std::io::Result<(u32, u32)> {
    use std::fs::File;
    use std::io::BufReader;

    let file = File::open(image_path)?;
    let mut reader = BufReader::new(file);

    // Read image dimensions using the image crate
    match image::io::Reader::new(&mut reader).with_guessed_format() {
        Ok(reader) => match reader.into_dimensions() {
            Ok((width, height)) => Ok((width, height)),
            Err(e) => {
                error!(
                    "Failed to read image dimensions for {}: {}",
                    image_path.display(),
                    e
                );
                Err(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
            }
        },
        Err(e) => {
            error!(
                "Failed to create image reader for {}: {}",
                image_path.display(),
                e
            );
            Err(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
        }
    }
}

/// Read and parse a single JSON file into an ImageAnnotation struct using streaming
/// This prevents OOM errors by parsing JSON directly from a file stream instead of
/// loading the entire file into memory first.
pub fn read_and_parse_json(path: &Path, buffer_size_kib: usize) -> Option<ImageAnnotation> {
    // Open the file as a stream
    let file = match fs::File::open(path) {
        Ok(file) => file,
        Err(e) => {
            error!("Failed to open JSON file ({}): {:?}", path.display(), e);
            return None;
        }
    };

    // Wrap file in BufReader with configurable capacity to reduce syscalls and improve throughput
    let reader = BufReader::with_capacity(buffer_size_kib * 1024, file);

    // Parse JSON from the buffered reader
    match serde_json::from_reader(reader) {
        Ok(annotation) => Some(annotation),
        Err(e) => {
            error!("Failed to parse JSON ({}): {:?}", path.display(), e);
            None
        }
    }
}

/// Read and parse a single JSON file into a StreamingImageAnnotation struct
/// This streams the image_data field to avoid materializing large strings in memory
pub fn read_and_parse_json_streaming(
    path: &Path,
    buffer_size_kib: usize,
) -> Option<StreamingImageAnnotation> {
    // Open the file as a stream
    let file = match fs::File::open(path) {
        Ok(file) => file,
        Err(e) => {
            error!("Failed to open JSON file ({}): {:?}", path.display(), e);
            return None;
        }
    };

    // Wrap file in BufReader with configurable capacity to reduce syscalls and improve throughput
    let reader = BufReader::with_capacity(buffer_size_kib * 1024, file);

    // Parse JSON from the buffered reader with streaming support for image_data
    match crate::streaming_json::read_and_parse_json_streaming(reader) {
        Ok(annotation) => Some(annotation),
        Err(e) => {
            error!(
                "Failed to parse JSON with streaming ({}): {:?}",
                path.display(),
                e
            );
            None
        }
    }
}

/// Read and parse a single JSON file into an ImageAnnotation struct
/// This function uses serde_json for parsing with buffered I/O
pub fn read_and_parse_json_buffered(
    path: &Path,
    buffer_size_kib: usize,
) -> Option<ImageAnnotation> {
    // Open the file as a stream
    let file = match fs::File::open(path) {
        Ok(file) => file,
        Err(e) => {
            error!("Failed to open JSON file ({}): {:?}", path.display(), e);
            return None;
        }
    };

    // Wrap file in BufReader with configurable capacity to reduce syscalls and improve throughput
    let mut reader = BufReader::with_capacity(buffer_size_kib * 1024, file);

    // Read the entire file into a buffer for processing
    let mut buffer = Vec::new();
    if let Err(e) = reader.read_to_end(&mut buffer) {
        error!("Failed to read JSON file ({}): {:?}", path.display(), e);
        return None;
    }

    // Parse using serde_json
    match serde_json::from_slice::<ImageAnnotation>(&buffer) {
        Ok(annotation) => Some(annotation),
        Err(e) => {
            error!("Failed to parse JSON ({}): {:?}", path.display(), e);
            None
        }
    }
}

/// Create a progress bar with the given length and label
pub fn create_progress_bar(len: u64, label: &str) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(&format!(
                "{{spinner:.green}} [{}] [{{elapsed_precise}}] [{{bar:40.cyan/blue}}] {{pos}}/{{len}} ({{eta}})",
                label
            ))
            .progress_chars("#>-"),
    );
    pb
}

/// Safely create output directories and return their paths
pub fn create_output_directory(path: &Path) -> std::io::Result<std::path::PathBuf> {
    if path.exists() {
        log::warn!(
            "Directory {:?} already exists. Deleting and recreating it.",
            path
        );
        fs::remove_dir_all(path).and_then(|_| fs::create_dir_all(path))?;
    } else {
        fs::create_dir_all(path)?;
    }
    Ok(path.to_path_buf())
}

/// Create a custom Rayon thread pool with limited concurrency
pub fn create_io_thread_pool(workers: usize) -> rayon::ThreadPool {
    let num_threads = std::cmp::min(workers, num_cpus::get_physical());

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .thread_name(|i| format!("labelme2yolo-io-{}", i))
        .build()
        .expect("Failed to create thread pool")
}

/// Generate a collision-resistant filename by combining the sanitized file stem
/// with a short hash of the relative path
pub fn generate_collision_resistant_name(file_stem: &str, relative_path: &Path) -> String {
    // Sanitize the file stem
    let sanitized_stem = sanitize_filename::sanitize(file_stem);

    // Generate a short hash of the relative path
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    relative_path.hash(&mut hasher);
    let hash = hasher.finish();

    // Use first 8 characters of hex representation for a short hash
    let short_hash = format!("{:x}", hash)[..8].to_string();

    // Combine sanitized stem with short hash
    format!("{}_{}", sanitized_stem, short_hash)
}

/// A simple semaphore implementation using standard library primitives
pub struct SimpleSemaphore {
    permits: Arc<(Mutex<usize>, Condvar)>,
    max_permits: usize,
}

impl SimpleSemaphore {
    /// Create a new semaphore with the given number of permits
    pub fn new(permits: usize) -> Self {
        Self {
            permits: Arc::new((Mutex::new(permits), Condvar::new())),
            max_permits: permits,
        }
    }

    /// Acquire a permit, blocking until one is available
    pub fn acquire(&self) {
        let (lock, cvar) = &*self.permits;
        let mut count = match lock.lock() {
            Ok(g) => g,
            Err(poisoned) => {
                log::warn!("SimpleSemaphore lock poisoned in acquire; continuing with inner value");
                poisoned.into_inner()
            }
        };
        while *count == 0 {
            count = match cvar.wait(count) {
                Ok(g) => g,
                Err(poisoned) => {
                    log::warn!(
                        "SimpleSemaphore lock poisoned after wait; continuing with inner value"
                    );
                    poisoned.into_inner()
                }
            };
        }
        *count -= 1;
    }

    /// Release a permit back to the semaphore
    pub fn release(&self) {
        let (lock, cvar) = &*self.permits;
        let mut count = match lock.lock() {
            Ok(g) => g,
            Err(poisoned) => {
                log::warn!("SimpleSemaphore lock poisoned in release; continuing with inner value");
                poisoned.into_inner()
            }
        };
        *count += 1;
        // Ensure we don't exceed the maximum permits
        if *count > self.max_permits {
            *count = self.max_permits;
        }
        cvar.notify_one();
    }

    /// Create a guard that automatically releases the permit when dropped
    pub fn acquire_guard(&self) -> SemaphoreGuard<'_> {
        self.acquire();
        SemaphoreGuard { semaphore: self }
    }
}

/// A guard that automatically releases the semaphore permit when dropped
pub struct SemaphoreGuard<'a> {
    semaphore: &'a SimpleSemaphore,
}

impl<'a> Drop for SemaphoreGuard<'a> {
    fn drop(&mut self) {
        self.semaphore.release();
    }
}
