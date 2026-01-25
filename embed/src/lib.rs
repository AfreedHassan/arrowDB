//! Arrow Embed - Rust library for text embeddings with C FFI
//!
//! Provides functions to embed text using all-MiniLM-L6-v2 model,
//! callable from C/C++.

pub mod dataset;

use std::collections::HashSet;
use std::ffi::{c_char, c_float, CStr};
use std::ptr;
use std::sync::Mutex;

use anyhow::Result;
use ndarray::{Array2, ArrayD, IxDyn};
use once_cell::sync::Lazy;
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

pub const EMBEDDING_DIM: usize = 384;

static EMBEDDER: Lazy<Mutex<Option<SentenceTransformer>>> = Lazy::new(|| Mutex::new(None));

#[repr(C)]
pub struct EmbeddingResult {
    pub data: *mut c_float,
    pub len: usize,
    pub error_code: i32,
}

pub struct SentenceTransformer {
    session: Session,
    tokenizer: Tokenizer,
}

impl SentenceTransformer {
    pub fn new(model_path: &str, tokenizer_name: &str) -> Result<Self> {
        let _ = ort::init().with_name("arrow_embed").commit();

        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("Failed to create session builder: {}", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("Failed to set optimization level: {}", e))?
            .with_intra_threads(4)
            .map_err(|e| anyhow::anyhow!("Failed to set thread count: {}", e))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow::anyhow!("Failed to load ONNX model: {}", e))?;

        let mut tokenizer = Tokenizer::from_pretrained(tokenizer_name, None)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: 384,
                ..Default::default()
            }))
            .map_err(|e| anyhow::anyhow!("Failed to set truncation: {}", e))?;

        tokenizer.with_padding(None);

        Ok(Self { session, tokenizer })
    }

    pub fn encode(&mut self, text: &str) -> Result<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();

        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&mask| mask as i64)
            .collect();

        let seq_len = input_ids.len();

        let input_ids_array = Array2::from_shape_vec((1, seq_len), input_ids)
            .map_err(|e| anyhow::anyhow!("Failed to create input_ids array: {}", e))?;

        let attention_mask_array = Array2::from_shape_vec((1, seq_len), attention_mask.clone())
            .map_err(|e| anyhow::anyhow!("Failed to create attention_mask array: {}", e))?;

        let output = self.run_inference(&input_ids_array, &attention_mask_array)?;

        // Check if output is already pooled (2D: [batch, hidden]) or raw tokens (3D: [batch, seq, hidden])
        let embedding = match output.shape().len() {
            2 => {
                // Already pooled by the model
                output.iter().copied().collect()
            }
            3 => {
                // Need to apply mean pooling
                mean_pooling(&output, &attention_mask_array)?
            }
            n => anyhow::bail!("Unexpected output tensor dimensions: {}D (expected 2D or 3D)", n),
        };

        Ok(normalize_l2(&embedding))
    }

    pub fn encode_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|text| self.encode(text)).collect()
    }

    fn run_inference(
        &mut self,
        input_ids: &Array2<i64>,
        attention_mask: &Array2<i64>,
    ) -> Result<ArrayD<f32>> {
        let input_names: HashSet<String> = self
            .session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();

        let input_ids_shape = input_ids.shape().to_vec();
        let (input_ids_data, _) = input_ids.clone().into_raw_vec_and_offset();
        let input_ids_tensor = Tensor::from_array((
            input_ids_shape.as_slice(),
            input_ids_data.into_boxed_slice(),
        ))
        .map_err(|e| anyhow::anyhow!("Failed to create input_ids tensor: {}", e))?;

        let attention_mask_shape = attention_mask.shape().to_vec();
        let (attention_mask_data, _) = attention_mask.clone().into_raw_vec_and_offset();
        let attention_mask_tensor = Tensor::from_array((
            attention_mask_shape.as_slice(),
            attention_mask_data.into_boxed_slice(),
        ))
        .map_err(|e| anyhow::anyhow!("Failed to create attention_mask tensor: {}", e))?;

        let outputs = if input_names.contains("token_type_ids") {
            let seq_len = input_ids.ncols();
            let token_type_ids_shape = vec![1, seq_len];
            let token_type_ids_data = vec![0i64; seq_len];
            let token_type_ids_tensor = Tensor::from_array((
                token_type_ids_shape.as_slice(),
                token_type_ids_data.into_boxed_slice(),
            ))
            .map_err(|e| anyhow::anyhow!("Failed to create token_type_ids tensor: {}", e))?;

            self.session
                .run(inputs![
                    "input_ids" => input_ids_tensor,
                    "attention_mask" => attention_mask_tensor,
                    "token_type_ids" => token_type_ids_tensor
                ])
                .map_err(|e| anyhow::anyhow!("Inference failed: {}", e))?
        } else {
            self.session
                .run(inputs![
                    "input_ids" => input_ids_tensor,
                    "attention_mask" => attention_mask_tensor
                ])
                .map_err(|e| anyhow::anyhow!("Inference failed: {}", e))?
        };

        // Try to extract last_hidden_state (output name varies by model)
        let output_tensor = if let Some(tensor) = outputs.get("last_hidden_state") {
            tensor
        } else if let Some(tensor) = outputs.get("token_embeddings") {
            tensor 
        } else {
            &outputs[0]
        };

        let (shape, data) = output_tensor
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to extract tensor: {}", e))?;

        let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let token_embeddings = ArrayD::from_shape_vec(IxDyn(&dims), data.to_vec())
            .map_err(|e| anyhow::anyhow!("Failed to create output array: {}", e))?;

        Ok(token_embeddings)
    }
}

fn mean_pooling(token_embeddings: &ArrayD<f32>, attention_mask: &Array2<i64>) -> Result<Vec<f32>> {
    let shape = token_embeddings.shape();
    anyhow::ensure!(shape.len() == 3, "Expected 3D tensor [batch, seq, hidden]");

    let (seq_len, hidden_dim) = (shape[1], shape[2]);
    let mut pooled = vec![0.0; hidden_dim];
    let mut sum_mask = 0.0;

    for i in 0..seq_len {
        if attention_mask[[0, i]] == 1 {
            for j in 0..hidden_dim {
                pooled[j] += token_embeddings[[0, i, j]];
            }
            sum_mask += 1.0;
        }
    }

    anyhow::ensure!(sum_mask > 0.0, "No tokens to pool");
    for val in pooled.iter_mut() {
        *val /= sum_mask;
    }

    Ok(pooled)
}

fn normalize_l2(embedding: &[f32]) -> Vec<f32> {
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm < 1e-12 {
        return embedding.to_vec();
    }

    embedding.iter().map(|x| x / norm).collect()
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ============================================================================
// C FFI Functions
// ============================================================================

#[unsafe(no_mangle)]
pub extern "C" fn arrow_embed_init(
    model_path: *const c_char,
    tokenizer_name: *const c_char,
) -> i32 {
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

    match SentenceTransformer::new(model_path_str, tokenizer_name_str) {
        Ok(embedder) => {
            *embedder_guard = Some(embedder);
            0
        }
        Err(e) => {
            eprintln!("Embedder initialization failed: {:#}", e);
            -5
        }
    }
}

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
                error_code: -4,
            };
        }
    };

    match embedder.encode(text_str) {
        Ok(embedding) => {
            let len = embedding.len();
            let mut boxed = embedding.into_boxed_slice();
            let data = boxed.as_mut_ptr();
            std::mem::forget(boxed);

            EmbeddingResult {
                data,
                len,
                error_code: 0,
            }
        }
        Err(e) => {
            eprintln!("Embedding failed: {:#}", e);
            EmbeddingResult {
                data: ptr::null_mut(),
                len: 0,
                error_code: -5,
            }
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn arrow_embed_free(result: EmbeddingResult) {
    if !result.data.is_null() && result.len > 0 {
        unsafe {
            let _ = Box::from_raw(std::slice::from_raw_parts_mut(result.data, result.len));
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn arrow_embed_dimension() -> usize {
    EMBEDDING_DIM
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentence_transformer_initialization() {
        let transformer = SentenceTransformer::new(
            "models/all-MiniLM-L6-v2.onnx",
            "sentence-transformers/all-MiniLM-L6-v2",
        );
        assert!(transformer.is_ok());
    }

    #[test]
    fn test_encode_returns_normalized_vector() {
        let mut transformer = SentenceTransformer::new(
            "models/all-MiniLM-L6-v2.onnx",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        .unwrap();

        let embedding = transformer.encode("Hello world").unwrap();

        assert_eq!(embedding.len(), 384);

        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "Embedding should be normalized");
    }

    #[test]
    fn test_cosine_similarity() {
        let mut transformer = SentenceTransformer::new(
            "models/all-MiniLM-L6-v2.onnx",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        .unwrap();

        let emb1 = transformer.encode("cat").unwrap();
        let emb2 = transformer.encode("dog").unwrap();
        let emb3 = transformer.encode("astronomy").unwrap();

        let sim_cat_dog = cosine_similarity(&emb1, &emb2);
        let sim_cat_astronomy = cosine_similarity(&emb1, &emb3);

        assert!(sim_cat_dog > sim_cat_astronomy);
    }

    #[test]
    fn test_batch_encoding() {
        let mut transformer = SentenceTransformer::new(
            "models/all-MiniLM-L6-v2.onnx",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        .unwrap();

        let texts = vec!["first", "second", "third"];
        let embeddings = transformer.encode_batch(&texts).unwrap();

        assert_eq!(embeddings.len(), 3);
        for emb in embeddings {
            assert_eq!(emb.len(), 384);
        }
    }
}
