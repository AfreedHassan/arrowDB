//! Dataset loading and processing for embedding generation
//!
//! Provides utilities to load OpenWebText dataset from HuggingFace,
//! process text chunks with tokenization, generate embeddings, and save to disk.

use std::ffi::{c_char, c_float, CStr};
use std::ptr;
use std::sync::Mutex;

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use once_cell::sync::Lazy;
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

/// Embedding dimension for all-MiniLM-L6-v2
pub const EMBEDDING_DIM: usize = 384;

/// Default paths
pub const DEFAULT_MODEL_PATH: &str = "models/all-MiniLM-L6-v2.onnx";
pub const DEFAULT_TOKENIZER_NAME: &str = "sentence-transformers/all-MiniLM-L6-v2";
pub const DEFAULT_COLLECTION_PATH: &str = "owt_collection";
pub const DEFAULT_TEXT_FILE: &str = "openwebtext.txt";
pub const DEFAULT_EMBEDDINGS_FILE: &str = "openwebtext-embeddings.bin";

/// Global dataset embedder instance (separate from query embedder for thread safety)
static DATASET_EMBEDDER: Lazy<Mutex<Option<DatasetEmbedder>>> = Lazy::new(|| Mutex::new(None));

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
            batch_size: 32,
        }
    }
}

/// Result of dataset loading containing text chunks and embeddings
pub struct DatasetResult {
    pub chunks: Vec<String>,
    pub embeddings: Vec<Vec<f32>>,
}

/// Internal embedder for dataset processing (mirrors lib.rs Embedder)
struct DatasetEmbedder {
    session: Session,
    tokenizer: Tokenizer,
}

impl DatasetEmbedder {
    fn new(model_path: &str, tokenizer_name: &str) -> Result<Self, String> {
        // Initialize ORT
        let _ = ort::init().with_name("arrow_dataset").commit();

        // Load model
        let session = Session::builder()
            .map_err(|e| format!("Failed to create session builder: {}", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("Failed to set optimization: {}", e))?
            .with_intra_threads(4)
            .map_err(|e| format!("Failed to set threads: {}", e))?
            .commit_from_file(model_path)
            .map_err(|e| format!("Failed to load model: {}", e))?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_pretrained(tokenizer_name, None)
            .map_err(|e| format!("Failed to load tokenizer: {}", e))?;

        Ok(DatasetEmbedder { session, tokenizer })
    }

    /// Embed a single text string (same logic as lib.rs)
    fn embed(&mut self, text: &str) -> Result<Vec<f32>, String> {
        // Tokenize
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| format!("Tokenization failed: {}", e))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&x| x as i64)
            .collect();
        let token_type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&x| x as i64).collect();
        let seq_len = input_ids.len();

        // Create tensors
        let input_ids_arr = Array2::from_shape_vec((1, seq_len), input_ids)
            .map_err(|e| format!("Failed to create input_ids array: {}", e))?;
        let attention_mask_arr = Array2::from_shape_vec((1, seq_len), attention_mask.clone())
            .map_err(|e| format!("Failed to create attention_mask array: {}", e))?;
        let token_type_ids_arr = Array2::from_shape_vec((1, seq_len), token_type_ids)
            .map_err(|e| format!("Failed to create token_type_ids array: {}", e))?;

        // Run inference
        let last_hidden_state =
            self.run_inference(input_ids_arr, attention_mask_arr.clone(), token_type_ids_arr)?;

        // Mean pooling
        let attention_mask_i64 = Array2::from_shape_vec((1, seq_len), attention_mask)
            .map_err(|e| format!("Failed to create mask array: {}", e))?;
        let pooled = mean_pooling(&last_hidden_state, &attention_mask_i64);

        // L2 normalize
        let normalized = normalize_l2(&pooled);

        // Return first (and only) row
        Ok(normalized.row(0).to_vec())
    }

    fn run_inference(
        &mut self,
        input_ids: Array2<i64>,
        attention_mask: Array2<i64>,
        token_type_ids: Array2<i64>,
    ) -> Result<ArrayD<f32>, String> {
        let input_ids_shape = input_ids.shape().to_vec();
        let (input_ids_data, _) = input_ids.into_raw_vec_and_offset();
        let input_ids_tensor =
            Tensor::from_array((input_ids_shape.as_slice(), input_ids_data.into_boxed_slice()))
                .map_err(|e| format!("Failed to create input_ids tensor: {}", e))?;

        let attention_mask_shape = attention_mask.shape().to_vec();
        let (attention_mask_data, _) = attention_mask.into_raw_vec_and_offset();
        let attention_mask_tensor = Tensor::from_array((
            attention_mask_shape.as_slice(),
            attention_mask_data.into_boxed_slice(),
        ))
        .map_err(|e| format!("Failed to create attention_mask tensor: {}", e))?;

        let token_type_ids_shape = token_type_ids.shape().to_vec();
        let (token_type_ids_data, _) = token_type_ids.into_raw_vec_and_offset();
        let token_type_ids_tensor = Tensor::from_array((
            token_type_ids_shape.as_slice(),
            token_type_ids_data.into_boxed_slice(),
        ))
        .map_err(|e| format!("Failed to create token_type_ids tensor: {}", e))?;

        let outputs = self
            .session
            .run(inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
                "token_type_ids" => token_type_ids_tensor
            ])
            .map_err(|e| format!("Inference failed: {}", e))?;

        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Failed to extract tensor: {}", e))?;

        let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        ArrayD::from_shape_vec(IxDyn(&dims), data.to_vec())
            .map_err(|e| format!("Failed to create output array: {}", e))
    }
}

/// Mean pooling over sequence dimension with attention mask
fn mean_pooling(last_hidden_state: &ArrayD<f32>, attention_mask: &Array2<i64>) -> Array2<f32> {
    let shape = last_hidden_state.shape();
    let (batch_size, seq_len, hidden_dim) = (shape[0], shape[1], shape[2]);

    let mut pooled = Array2::<f32>::zeros((batch_size, hidden_dim));

    for b in 0..batch_size {
        let mut sum = Array1::<f32>::zeros(hidden_dim);
        let mut count = 0.0f32;

        for s in 0..seq_len {
            let mask_val = attention_mask[[b, s]] as f32;
            if mask_val > 0.0 {
                for h in 0..hidden_dim {
                    sum[h] += last_hidden_state[[b, s, h]] * mask_val;
                }
                count += mask_val;
            }
        }

        if count > 0.0 {
            for h in 0..hidden_dim {
                pooled[[b, h]] = sum[h] / count;
            }
        }
    }

    pooled
}

/// L2 normalize embeddings
fn normalize_l2(embeddings: &Array2<f32>) -> Array2<f32> {
    let mut normalized = embeddings.clone();
    let (batch_size, dim) = (embeddings.nrows(), embeddings.ncols());

    for b in 0..batch_size {
        let mut norm = 0.0f32;
        for d in 0..dim {
            norm += embeddings[[b, d]].powi(2);
        }
        norm = norm.sqrt();

        if norm > 1e-12 {
            for d in 0..dim {
                normalized[[b, d]] = embeddings[[b, d]] / norm;
            }
        }
    }

    normalized
}

// ============================================================================
// Dataset Processing Functions
// ============================================================================

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

/// OpenWebText item structure for deserialization from HuggingFace
#[derive(serde::Deserialize, Debug, Clone)]
struct OpenWebTextItem {
    pub text: String,
}

/// Sentence tokenization using unicode-segmentation (similar to Python's NLTK sent_tokenize)
/// Uses Unicode Standard Annex #29 sentence boundaries
fn tokenize_sentences(text: &str) -> Vec<String> {
    use unicode_segmentation::UnicodeSegmentation;

    text.unicode_sentences()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Download OpenWebText from HuggingFace using burn-dataset and extract text samples
/// Similar to Python: ds = load_dataset("openwebtext", split="train", streaming=True)
fn download_openwebtext(num_samples: usize) -> Result<Vec<String>, String> {
    use burn_dataset::Dataset;
    use burn_dataset::HuggingfaceDatasetLoader;

    eprintln!("Loading OpenWebText from HuggingFace using burn-dataset...");
    eprintln!("This requires Python with 'datasets' library installed.");
    eprintln!("Note: First run will download the dataset (~5-10GB compressed)");

    // Load OpenWebText dataset using HuggingfaceDatasetLoader
    // Config is 'plain_text' (the only available config)
    let dataset: burn_dataset::SqliteDataset<OpenWebTextItem> =
        HuggingfaceDatasetLoader::new("openwebtext")
            .dataset("train")
            .take(200_000)
            .map_err(|e| format!("Failed to load OpenWebText dataset: {}", e))?;

    eprintln!("Dataset loaded. Processing {} items...", dataset.len());

    let mut chunks = Vec::new();
    let min_length = 40;
    let max_length = 200;

    // Process dataset similar to Python:
    // for row in ds:
    //     for c in sent_tokenize(row["text"]):
    //         if 40 <= len(c) <= 200:
    //             chunks.append(c)
    for i in 0..dataset.len() {
        if chunks.len() >= num_samples {
            break;
        }

        if let Some(item) = dataset.get(i) {
            // Sentence tokenization (like Python's sent_tokenize)
            for sentence in tokenize_sentences(&item.text) {
                let len = sentence.len();
                if len >= min_length && len <= max_length {
                    chunks.push(sentence);
                    if chunks.len() >= num_samples {
                        break;
                    }
                }
            }
        }

        // Progress update every 1000 items
        if i % 1000 == 0 && i > 0 {
            eprint!("\rProcessed {}/{} documents, collected {} chunks", i, dataset.len(), chunks.len());
        }
    }
    eprintln!("\rProcessed documents, collected {} chunks          ", chunks.len());

    if chunks.is_empty() {
        return Err("No chunks extracted from dataset".to_string());
    }

    // Truncate to exact number requested (like Python: chunks = chunks[:200000])
    chunks.truncate(num_samples);

    eprintln!("Extracted {} text chunks", chunks.len());
    Ok(chunks)
}

/// Generate embeddings for a batch of texts
fn embed_texts_batch(
    embedder: &mut DatasetEmbedder,
    texts: &[String],
    progress_callback: Option<&dyn Fn(usize, usize)>,
) -> Result<Vec<Vec<f32>>, String> {
    let mut embeddings = Vec::with_capacity(texts.len());

    for (i, text) in texts.iter().enumerate() {
        let embedding = embedder.embed(text)?;
        embeddings.push(embedding);

        if let Some(callback) = progress_callback {
            if i % 100 == 0 || i == texts.len() - 1 {
                callback(i + 1, texts.len());
            }
        }
    }

    Ok(embeddings)
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

/// Download OpenWebText from HuggingFace, generate embeddings, and return to C++
///
/// This function downloads text from HuggingFace, embeds using MiniLM,
/// and returns the data to C++ for collection creation.
/// C++ is responsible for creating and saving the collection.
#[unsafe(no_mangle)]
pub extern "C" fn arrow_dataset_download_and_embed(
    model_path: *const c_char,
    tokenizer_name: *const c_char,
    num_samples: usize,
    output_text_path: *const c_char,
) -> DatasetLoadResult {
    let model_path_str = if model_path.is_null() {
        DEFAULT_MODEL_PATH.to_string()
    } else {
        match unsafe { CStr::from_ptr(model_path) }.to_str() {
            Ok(s) => s.to_string(),
            Err(_) => {
                return DatasetLoadResult {
                    embeddings_ptr: ptr::null_mut(),
                    chunks_ptr: ptr::null_mut(),
                    num_chunks: 0,
                    embedding_dim: EMBEDDING_DIM,
                    error_code: -1,
                }
            }
        }
    };

    let tokenizer_name_str = if tokenizer_name.is_null() {
        DEFAULT_TOKENIZER_NAME.to_string()
    } else {
        match unsafe { CStr::from_ptr(tokenizer_name) }.to_str() {
            Ok(s) => s.to_string(),
            Err(_) => {
                return DatasetLoadResult {
                    embeddings_ptr: ptr::null_mut(),
                    chunks_ptr: ptr::null_mut(),
                    num_chunks: 0,
                    embedding_dim: EMBEDDING_DIM,
                    error_code: -2,
                }
            }
        }
    };

    // Initialize embedder
    eprintln!("Initializing embedder...");
    let mut embedder = match DatasetEmbedder::new(&model_path_str, &tokenizer_name_str) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Failed to initialize embedder: {}", e);
            return DatasetLoadResult {
                embeddings_ptr: ptr::null_mut(),
                chunks_ptr: ptr::null_mut(),
                num_chunks: 0,
                embedding_dim: EMBEDDING_DIM,
                error_code: -3,
            };
        }
    };

    // Download OpenWebText
    let texts = match download_openwebtext(num_samples) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to download OpenWebText: {}", e);
            return DatasetLoadResult {
                embeddings_ptr: ptr::null_mut(),
                chunks_ptr: ptr::null_mut(),
                num_chunks: 0,
                embedding_dim: EMBEDDING_DIM,
                error_code: -4,
            };
        }
    };

    if texts.is_empty() {
        eprintln!("No texts downloaded");
        return DatasetLoadResult {
            embeddings_ptr: ptr::null_mut(),
            chunks_ptr: ptr::null_mut(),
            num_chunks: 0,
            embedding_dim: EMBEDDING_DIM,
            error_code: -5,
        };
    }

    eprintln!("Generating embeddings for {} texts...", texts.len());

    // Generate embeddings with progress
    let embeddings = match embed_texts_batch(&mut embedder, &texts, Some(&|current, total| {
        eprint!(
            "\rEmbedding progress: {}/{} ({:.1}%)",
            current,
            total,
            (current as f64 / total as f64) * 100.0
        );
    })) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("\nFailed to generate embeddings: {}", e);
            return DatasetLoadResult {
                embeddings_ptr: ptr::null_mut(),
                chunks_ptr: ptr::null_mut(),
                num_chunks: 0,
                embedding_dim: EMBEDDING_DIM,
                error_code: -6,
            };
        }
    };
    eprintln!(); // Newline after progress

    // Save text file if output path provided (like Python embed.py)
    if !output_text_path.is_null() {
        if let Ok(path_str) = unsafe { CStr::from_ptr(output_text_path) }.to_str() {
            if let Err(e) = save_text_file(&texts, path_str) {
                eprintln!("Warning: Failed to save text file: {}", e);
            } else {
                eprintln!("Saved text file to: {}", path_str);
            }
        }
    }

    let loaded_count = texts.len();

    // Convert embeddings to C array (flat float32 array)
    let mut flat_embeddings = Vec::new();
    for embedding in &embeddings {
        flat_embeddings.extend_from_slice(embedding);
    }

    let mut flat_embeddings_boxed = flat_embeddings.into_boxed_slice();
    let embeddings_ptr = flat_embeddings_boxed.as_mut_ptr();
    std::mem::forget(flat_embeddings_boxed); // Caller must free

    // Convert chunks to C string array
    let chunks_vec: Vec<*mut c_char> = texts
        .iter()
        .map(|chunk| {
            let c_string = std::ffi::CString::new(chunk.clone()).unwrap();
            c_string.into_raw()
        })
        .collect();

    let mut chunks_boxed = chunks_vec.into_boxed_slice();
    let chunks_ptr = chunks_boxed.as_mut_ptr();
    std::mem::forget(chunks_boxed); // Caller must free

    eprintln!("Returning {} chunks with embeddings to C++", loaded_count);

    DatasetLoadResult {
        embeddings_ptr,
        chunks_ptr,
        num_chunks: loaded_count,
        embedding_dim: EMBEDDING_DIM,
        error_code: 0,
    }
}

/// Load OpenWebText dataset from text and embeddings files.
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
            embedding_dim: EMBEDDING_DIM,
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
                embedding_dim: EMBEDDING_DIM,
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
                embedding_dim: EMBEDDING_DIM,
                error_code: -3,
            }
        }
    };

    // Load text chunks
    let chunks = match load_text_chunks_from_file(text_path_str, min_length, max_length, num_chunks)
    {
        Ok(c) => c,
        Err(_) => {
            return DatasetLoadResult {
                embeddings_ptr: ptr::null_mut(),
                chunks_ptr: ptr::null_mut(),
                num_chunks: 0,
                embedding_dim: EMBEDDING_DIM,
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
            embedding_dim: EMBEDDING_DIM,
            error_code: -5,
        };
    }

    // Load embeddings from binary file
    let embeddings = match load_embeddings_from_file(embeddings_path_str, loaded_count, EMBEDDING_DIM) {
        Ok(e) => e,
        Err(_) => {
            return DatasetLoadResult {
                embeddings_ptr: ptr::null_mut(),
                chunks_ptr: ptr::null_mut(),
                num_chunks: 0,
                embedding_dim: EMBEDDING_DIM,
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
            c_string.into_raw()
        })
        .collect();

    let mut chunks_boxed = chunks_vec.into_boxed_slice();
    let chunks_ptr = chunks_boxed.as_mut_ptr();
    std::mem::forget(chunks_boxed); // Caller must free

    DatasetLoadResult {
        embeddings_ptr,
        chunks_ptr,
        num_chunks: loaded_count,
        embedding_dim: EMBEDDING_DIM,
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
            let chunks_slice =
                std::slice::from_raw_parts_mut(result.chunks_ptr, result.num_chunks);
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
