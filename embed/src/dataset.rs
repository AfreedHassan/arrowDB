//! Dataset loading and processing for embedding generation
//!
//! Provides utilities to load OpenWebText dataset from Parquet files,
//! process text chunks with tokenization and filtering, and generate embeddings.

use std::ffi::{c_char, c_float, CStr};
use std::ptr;
use unicode_segmentation::UnicodeSegmentation;

/// Configuration for dataset loading
pub struct DatasetConfig {
    pub min_length: usize,
    pub max_length: usize,
    pub num_chunks: usize,
    pub batch_size: usize,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            min_length: 40,
            max_length: 200,
            num_chunks: 200_000,
            batch_size: 128,
        }
    }
}

/// Result of dataset loading containing text chunks and embeddings
pub struct DatasetResult {
    pub chunks: Vec<String>,
    pub embeddings: Vec<Vec<f32>>,
}

// TODO: use tokenizers::Tokenizers
pub fn tokenize_sentences(text: &str) -> Vec<String> {
    let v : Vec<String> = Vec::new();
    v
}

/// Filter sentences by length (keep only sentences within min/max length)
pub fn filter_by_length(sentences: Vec<String>, min_len: usize, max_len: usize) -> Vec<String> {
    sentences
        .into_iter()
        .filter(|s| s.len() >= min_len && s.len() <= max_len)
        .collect()
}

/// Save dataset chunks to text file (one chunk per line)
pub fn save_text_file(chunks: &[String], output_path: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(output_path)?;
    for chunk in chunks {
        writeln!(file, "{}", chunk.replace("\n", " "))?;
    }
    Ok(())
}

/// Save embeddings to binary file (flat array of float32)
pub fn save_embeddings_file(embeddings: &[Vec<f32>], output_path: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(output_path)?;

    for embedding in embeddings {
        for &value in embedding {
            file.write_all(&value.to_le_bytes())?;
        }
    }
    Ok(())
}

/// Save vector IDs to binary file (uint64 array)
pub fn save_ids_file(num_chunks: usize, output_path: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(output_path)?;

    for i in 0..num_chunks {
        let id = i as u64;
        file.write_all(&id.to_le_bytes())?;
    }
    Ok(())
}

/// Load text chunks from a plain text file (one chunk per line)
pub fn load_text_chunks_from_file(
    path: &str,
    min_len: usize,
    max_len: usize,
    max_chunks: usize,
) -> std::io::Result<Vec<String>> {
    use std::io::{BufRead, BufReader};

    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);

    let mut chunks = Vec::new();
    for line in reader.lines() {
        if chunks.len() >= max_chunks {
            break;
        }
        let line = line?;
        let trimmed = line.trim();
        if trimmed.len() >= min_len && trimmed.len() <= max_len {
            chunks.push(trimmed.to_string());
        }
    }

    Ok(chunks)
}

/// Process text data: tokenize sentences and filter by length
pub fn process_text_data(
    text: &str,
    config: &DatasetConfig,
) -> Vec<String> {
    let sentences = tokenize_sentences(text);
    let filtered = filter_by_length(sentences, config.min_length, config.max_length);
    filtered
        .into_iter()
        .take(config.num_chunks)
        .collect()
}

// ============================================================================
// C FFI Types and Functions
// ============================================================================

/// Result structure returned to C/C++ from dataset loading
#[repr(C)]
pub struct DatasetLoadResult {
    /// Flat array of embeddings (num_chunks * embedding_dim * sizeof(float))
    pub embeddings_ptr: *mut c_float,
    /// Array of C strings (num_chunks pointers to null-terminated strings)
    pub chunks_ptr: *mut *mut c_char,
    /// Number of chunks loaded
    pub num_chunks: usize,
    /// Embedding dimension (384 for MiniLM)
    pub embedding_dim: usize,
    /// Error code: 0 = success, non-zero = error
    pub error_code: i32,
}

/// Load OpenWebText dataset from text and embeddings files.
///
/// Reads from two files:
/// - text_path: Plain text file, one chunk per line
/// - embeddings_path: Binary file with float32 embeddings (must be num_chunks * 384 floats)
///
/// # Arguments
/// * `text_path` - Path to text file
/// * `embeddings_path` - Path to binary embeddings file
/// * `num_chunks` - Number of chunks to load
/// * `min_length` - Minimum text length (characters)
/// * `max_length` - Maximum text length (characters)
///
/// # Returns
/// DatasetLoadResult with loaded data or error code
#[unsafe(no_mangle)]
pub extern "C" fn arrow_dataset_load_openwebtext(
    text_path: *const c_char,
    embeddings_path: *const c_char,
    num_chunks: usize,
    min_length: usize,
    max_length: usize,
) -> DatasetLoadResult {
    // Validate inputs
    if text_path.is_null() || embeddings_path.is_null() {
        return DatasetLoadResult {
            embeddings_ptr: ptr::null_mut(),
            chunks_ptr: ptr::null_mut(),
            num_chunks: 0,
            embedding_dim: 384,
            error_code: -1,
        };
    }

    let text_path_str = match unsafe { CStr::from_ptr(text_path) }.to_str() {
        Ok(s) => s,
        Err(_) => {
            return DatasetLoadResult {
                embeddings_ptr: ptr::null_mut(),
                chunks_ptr: ptr::null_mut(),
                num_chunks: 0,
                embedding_dim: 384,
                error_code: -2,
            }
        }
    };

    let embeddings_path_str = match unsafe { CStr::from_ptr(embeddings_path) }.to_str() {
        Ok(s) => s,
        Err(_) => {
            return DatasetLoadResult {
                embeddings_ptr: ptr::null_mut(),
                chunks_ptr: ptr::null_mut(),
                num_chunks: 0,
                embedding_dim: 384,
                error_code: -3,
            }
        }
    };

    // Load text chunks
    let chunks = match load_text_chunks_from_file(text_path_str, min_length, max_length, num_chunks) {
        Ok(c) => c,
        Err(_) => {
            return DatasetLoadResult {
                embeddings_ptr: ptr::null_mut(),
                chunks_ptr: ptr::null_mut(),
                num_chunks: 0,
                embedding_dim: 384,
                error_code: -4,
            }
        }
    };

    let loaded_count = chunks.len();
    if loaded_count == 0 {
        return DatasetLoadResult {
            embeddings_ptr: ptr::null_mut(),
            chunks_ptr: ptr::null_mut(),
            num_chunks: 0,
            embedding_dim: 384,
            error_code: -5,
        };
    }

    // Load embeddings from binary file
    let embeddings = match load_embeddings_from_file(embeddings_path_str, loaded_count, 384) {
        Ok(e) => e,
        Err(_) => {
            return DatasetLoadResult {
                embeddings_ptr: ptr::null_mut(),
                chunks_ptr: ptr::null_mut(),
                num_chunks: 0,
                embedding_dim: 384,
                error_code: -6,
            }
        }
    };

    // Convert embeddings to C array (flat float32 array)
    let mut flat_embeddings = Vec::new();
    for embedding in &embeddings {
        flat_embeddings.extend_from_slice(embedding);
    }

    let mut flat_embeddings_boxed = flat_embeddings.into_boxed_slice();
    let embeddings_ptr = flat_embeddings_boxed.as_mut_ptr();
    std::mem::forget(flat_embeddings_boxed); // Caller must free

    // Convert chunks to C string array
    let chunks_vec: Vec<*mut c_char> = chunks
        .iter()
        .map(|chunk| {
            let c_string = std::ffi::CString::new(chunk.clone()).unwrap();
            let ptr = c_string.into_raw();
            ptr
        })
        .collect();

    let mut chunks_boxed = chunks_vec.into_boxed_slice();
    let chunks_ptr = chunks_boxed.as_mut_ptr();
    std::mem::forget(chunks_boxed); // Caller must free

    DatasetLoadResult {
        embeddings_ptr,
        chunks_ptr,
        num_chunks: loaded_count,
        embedding_dim: 384,
        error_code: 0,
    }
}

/// Free a dataset load result
#[unsafe(no_mangle)]
pub extern "C" fn arrow_dataset_free(result: DatasetLoadResult) {
    if !result.embeddings_ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(std::slice::from_raw_parts_mut(
                result.embeddings_ptr,
                result.num_chunks * result.embedding_dim,
            ));
        }
    }

    if !result.chunks_ptr.is_null() {
        unsafe {
            let chunks_slice = std::slice::from_raw_parts_mut(result.chunks_ptr, result.num_chunks);
            // Free each C string
            for ptr in chunks_slice.iter() {
                if !ptr.is_null() {
                    let _ = std::ffi::CString::from_raw(*ptr);
                }
            }
            // Free the array itself
            let _ = Box::from_raw(chunks_slice.as_mut_ptr());
        }
    }
}

// ============================================================================
// Private Helper Functions
// ============================================================================

/// Load embeddings from binary file (little-endian float32)
fn load_embeddings_from_file(
    path: &str,
    num_embeddings: usize,
    embedding_dim: usize,
) -> std::io::Result<Vec<Vec<f32>>> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)?;
    let mut embeddings = Vec::with_capacity(num_embeddings);

    for _ in 0..num_embeddings {
        let mut embedding = Vec::with_capacity(embedding_dim);
        for _ in 0..embedding_dim {
            let mut bytes = [0u8; 4];
            file.read_exact(&mut bytes)?;
            let value = f32::from_le_bytes(bytes);
            embedding.push(value);
        }
        embeddings.push(embedding);
    }

    Ok(embeddings)
}
