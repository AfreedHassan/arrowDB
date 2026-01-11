//! Arrow Embed - Rust library for text embeddings with C FFI
//!
//! Provides functions to embed text using all-MiniLM-L6-v2 model,
//! callable from C/C++.

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

/// Global embedder instance (lazy initialized)
static EMBEDDER: Lazy<Mutex<Option<Embedder>>> = Lazy::new(|| Mutex::new(None));

/// Result returned to C/C++ containing the embedding vector
#[repr(C)]
pub struct EmbeddingResult {
    /// Pointer to embedding data (caller must free with free_embedding)
    pub data: *mut c_float,
    /// Length of the embedding vector (384 for MiniLM)
    pub len: usize,
    /// Error code: 0 = success, non-zero = error
    pub error_code: i32,
}

/// Internal embedder holding the model and tokenizer
struct Embedder {
    session: Session,
    tokenizer: Tokenizer,
}

impl Embedder {
    fn new(model_path: &str, tokenizer_name: &str) -> Result<Self, String> {
        // Initialize ORT
        let _ = ort::init().with_name("arrow_embed").commit();

        // Load model
        let session = Session::builder()
            .map_err(|e| format!("Failed to create session builder: {}", e))? 
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("Failed to set optimization: {}", e))?
            .with_intra_threads(4)
            .map_err(|e| format!("Failed to set threads: {}", e))?
            .commit_from_file(model_path)
            .map_err(|e| format!("Failed to load model: {}", e))?;
        // map_err expects a error handler 
        // |e| is closure aka lambda capture group in cpp terms
        // the part after |e| is the lambda body
        // each line between a map_err is setting up params/opts for the session

        // Load tokenizer
        let tokenizer = Tokenizer::from_pretrained(tokenizer_name, None)
            .map_err(|e| format!("Failed to load tokenizer: {}", e))?;

        Ok(Embedder { session, tokenizer })
    }

    fn embed(&mut self, text: &str) -> Result<Vec<f32>, String> {
        // Tokenize
        let encoding = self.tokenizer
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
// C FFI Functions
// ============================================================================

/// Initialize the embedder with model and tokenizer paths.
/// Must be called before embed_text().
///
/// # Arguments
/// * `model_path` - Path to the ONNX model file (e.g., "models/all-MiniLM-L6-v2.onnx")
/// * `tokenizer_name` - HuggingFace tokenizer name (e.g., "sentence-transformers/all-MiniLM-L6-v2")
///
/// # Returns
/// * 0 on success, non-zero error code on failure
#[unsafe(no_mangle)]
pub extern "C" fn arrow_embed_init( model_path: *const c_char, tokenizer_name: *const c_char) -> i32 {
    if model_path.is_null() || tokenizer_name.is_null() {
        return -1;
    }

    let model_path_str = match unsafe { CStr::from_ptr(model_path) }.to_str() {
        Ok(s) => s,
        Err(_) => return -2,
    };

    let tokenizer_name_str = match unsafe { CStr::from_ptr(tokenizer_name) }.to_str() {
        Ok(s) => s,
        Err(_) => return -3,
    };

    let mut embedder_guard = match EMBEDDER.lock() {
        Ok(g) => g,
        Err(_) => return -4,
    };

    match Embedder::new(model_path_str, tokenizer_name_str) {
        Ok(embedder) => {
            *embedder_guard = Some(embedder);
            0
        }
        Err(_) => -5,
    }
}

/// Embed a text string and return the embedding vector.
///
/// # Arguments
/// * `text` - Null-terminated C string to embed
///
/// # Returns
/// * EmbeddingResult containing pointer to float array, length, and error code
/// * Caller must free the data pointer using free_embedding()
#[unsafe(no_mangle)]
pub extern "C" fn arrow_embed_text(text: *const c_char) -> EmbeddingResult {
    if text.is_null() {
        return EmbeddingResult {
            data: ptr::null_mut(),
            len: 0,
            error_code: -1,
        };
    }

    let text_str = match unsafe { CStr::from_ptr(text) }.to_str() {
        Ok(s) => s,
        Err(_) => {
            return EmbeddingResult {
                data: ptr::null_mut(),
                len: 0,
                error_code: -2,
            }
        }
    };

    let mut embedder_guard = match EMBEDDER.lock() {
        Ok(g) => g,
        Err(_) => {
            return EmbeddingResult {
                data: ptr::null_mut(),
                len: 0,
                error_code: -3,
            }
        }
    };

    let embedder = match embedder_guard.as_mut() {
        Some(e) => e,
        None => {
            return EmbeddingResult {
                data: ptr::null_mut(),
                len: 0,
                error_code: -4, // Not initialized
            }
        }
    };

    match embedder.embed(text_str) {
        Ok(embedding) => {
            let len = embedding.len();
            let mut boxed = embedding.into_boxed_slice();
            let data = boxed.as_mut_ptr();
            std::mem::forget(boxed); // Prevent deallocation, caller must free

            EmbeddingResult {
                data,
                len,
                error_code: 0,
            }
        }
        Err(_) => EmbeddingResult {
            data: ptr::null_mut(),
            len: 0,
            error_code: -5,
        },
    }
}

/// Free an embedding result allocated by embed_text().
///
/// # Arguments
/// * `result` - The EmbeddingResult to free
#[unsafe(no_mangle)]
pub extern "C" fn arrow_embed_free(result: EmbeddingResult) {
    if !result.data.is_null() && result.len > 0 {
        unsafe {
            // Reconstruct the Box and let it drop
            let _ = Box::from_raw(std::slice::from_raw_parts_mut(result.data, result.len));
        }
    }
}

/// Get the embedding dimension (384 for all-MiniLM-L6-v2).
#[unsafe(no_mangle)]
pub extern "C" fn arrow_embed_dimension() -> usize {
    EMBEDDING_DIM
}
