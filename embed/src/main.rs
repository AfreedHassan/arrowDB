use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::inputs;
use ort::value::Tensor;
use anyhow::{Result, Context, anyhow};
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use std::path::Path;
use tokenizers::Tokenizer;

/// Initialize ONNX Runtime and load model from path
fn load_model<P: AsRef<Path>>(model_path: P) -> Result<Session> {
    // Initialize ORT (only needs to be done once globally, returns bool)
    let _ = ort::init()
        .with_name("arrow_embeddings")
        .commit();

    // Build session from file
    Session::builder()
        .context("Failed to create session builder")?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .context("Failed to set optimization level")?
        .with_intra_threads(4)
        .context("Failed to set intra-op threads")?
        .commit_from_file(model_path)
        .context("Failed to load ONNX model")
}

/// Run inference with input_ids, attention_mask, and token_type_ids (transformer model)
fn run_transformer_inference(
    session: &mut Session,
    input_ids: Array2<i64>,
    attention_mask: Array2<i64>,
    token_type_ids: Array2<i64>,
) -> Result<ArrayD<f32>> {
    // Create tensors from ndarrays using (shape, data) tuple format
    let input_ids_shape = input_ids.shape().to_vec();
    let (input_ids_data, _) = input_ids.into_raw_vec_and_offset();
    let input_ids_tensor = Tensor::from_array((input_ids_shape.as_slice(), input_ids_data.into_boxed_slice()))?;

    let attention_mask_shape = attention_mask.shape().to_vec();
    let (attention_mask_data, _) = attention_mask.into_raw_vec_and_offset();
    let attention_mask_tensor = Tensor::from_array((attention_mask_shape.as_slice(), attention_mask_data.into_boxed_slice()))?;

    let token_type_ids_shape = token_type_ids.shape().to_vec();
    let (token_type_ids_data, _) = token_type_ids.into_raw_vec_and_offset();
    let token_type_ids_tensor = Tensor::from_array((token_type_ids_shape.as_slice(), token_type_ids_data.into_boxed_slice()))?;

    // Run inference with named inputs (inputs! returns Vec for named inputs)
    let outputs = session
        .run(inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
            "token_type_ids" => token_type_ids_tensor
        ])
        .context("Failed to run inference")?;

    // Extract output tensor - returns (&Shape, &[T])
    let (shape, data) = outputs[0]
        .try_extract_tensor::<f32>()
        .context("Failed to extract output tensor")?;

    // Convert to owned ndarray
    let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    let output = ArrayD::from_shape_vec(IxDyn(&dims), data.to_vec())
        .context("Failed to create ndarray from output")?;

    Ok(output)
}

/// Mean pooling for embeddings (common for sentence embeddings)
/// Takes last_hidden_state [batch, seq_len, hidden_dim] and attention_mask [batch, seq_len]
#[allow(dead_code)]
fn mean_pooling(
    last_hidden_state: &ArrayD<f32>,
    attention_mask: &Array2<i64>,
) -> Array2<f32> {
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

/// L2 normalize embeddings (for cosine similarity)
#[allow(dead_code)]
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


fn main() -> Result<()> {
    // Load tokenizer
    let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None)
        .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

    let encoding = tokenizer.encode("The quick bronwn fox jumps over the lazy dog.", false)
        .map_err(|e| anyhow!("Failed to encode text: {}", e))?;
    println!("Tokens: {:?}", encoding.get_tokens());

    // Load model
    let mut session = load_model("models/all-MiniLM-L6-v2.onnx")?;

    // Print input/output info
    println!("\nModel inputs:");
    for (i, input) in session.inputs().iter().enumerate() {
        println!("  [{}] name: {:?}, dtype: {:?}",
            i, input.name(), input.dtype());
    }

    println!("\nModel outputs:");
    for (i, output) in session.outputs().iter().enumerate() {
        println!("  [{}] name: {:?}, dtype: {:?}",
            i, output.name(), output.dtype());
    }

    // Example: Run inference with tokenized input
    let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
    let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();
    let token_type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&x| x as i64).collect();
    let seq_len = input_ids.len();

    let input_ids_arr = Array2::from_shape_vec((1, seq_len), input_ids)?;
    let attention_mask_arr = Array2::from_shape_vec((1, seq_len), attention_mask)?;
    let token_type_ids_arr = Array2::from_shape_vec((1, seq_len), token_type_ids)?;

    let output = run_transformer_inference(&mut session, input_ids_arr, attention_mask_arr, token_type_ids_arr)?;
    println!("\nOutput shape: {:?}", output.shape());

    Ok(())
}
