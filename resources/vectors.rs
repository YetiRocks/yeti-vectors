use yeti_core::prelude::*;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

// ============================================================================
// Model Cache Directory (dylib-local static — set from on_ready)
// ============================================================================

fn models_dir() -> &'static OnceLock<PathBuf> {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    &DIR
}

fn get_models_dir() -> PathBuf {
    models_dir()
        .get()
        .cloned()
        .unwrap_or_else(|| PathBuf::from(".fastembed_cache"))
}

// ============================================================================
// Model Caches (dylib-local statics — NOT shared with host)
// ============================================================================

fn text_models() -> &'static DashMap<String, Arc<Mutex<fastembed::TextEmbedding>>> {
    static CACHE: OnceLock<DashMap<String, Arc<Mutex<fastembed::TextEmbedding>>>> = OnceLock::new();
    CACHE.get_or_init(DashMap::new)
}

fn image_models() -> &'static DashMap<String, Arc<Mutex<fastembed::ImageEmbedding>>> {
    static CACHE: OnceLock<DashMap<String, Arc<Mutex<fastembed::ImageEmbedding>>>> = OnceLock::new();
    CACHE.get_or_init(DashMap::new)
}

// ============================================================================
// Model Loading Helpers
// ============================================================================

fn get_or_init_text_model(model_name: &str) -> std::result::Result<Arc<Mutex<fastembed::TextEmbedding>>, String> {
    if let Some(entry) = text_models().get(model_name) {
        return Ok(entry.value().clone());
    }

    let cache_dir = get_models_dir();
    eprintln!("[yeti-vectors] Initializing text model: {} (cache: {})", model_name, cache_dir.display());

    let model = fastembed::TextEmbedding::try_new(
        fastembed::InitOptions::new(
            parse_text_model(model_name)
        )
        .with_cache_dir(cache_dir)
        .with_show_download_progress(true)
    ).map_err(|e| format!("Failed to init text model '{}': {}", model_name, e))?;

    let arc = Arc::new(Mutex::new(model));
    text_models().insert(model_name.to_string(), arc.clone());
    eprintln!("[yeti-vectors] Text model '{}' ready", model_name);
    Ok(arc)
}

fn get_or_init_image_model(model_name: &str) -> std::result::Result<Arc<Mutex<fastembed::ImageEmbedding>>, String> {
    if let Some(entry) = image_models().get(model_name) {
        return Ok(entry.value().clone());
    }

    let cache_dir = get_models_dir();
    eprintln!("[yeti-vectors] Initializing image model: {} (cache: {})", model_name, cache_dir.display());

    let model = fastembed::ImageEmbedding::try_new(
        fastembed::ImageInitOptions::new(
            parse_image_model(model_name)
        )
        .with_cache_dir(cache_dir)
        .with_show_download_progress(true)
    ).map_err(|e| format!("Failed to init image model '{}': {}", model_name, e))?;

    let arc = Arc::new(Mutex::new(model));
    image_models().insert(model_name.to_string(), arc.clone());
    eprintln!("[yeti-vectors] Image model '{}' ready", model_name);
    Ok(arc)
}

// ============================================================================
// Model Name Parsing
// ============================================================================

fn parse_text_model(name: &str) -> fastembed::EmbeddingModel {
    match name {
        "BAAI/bge-small-en-v1.5" | "bge-small-en-v1.5" => fastembed::EmbeddingModel::BGESmallENV15,
        "BAAI/bge-base-en-v1.5" | "bge-base-en-v1.5" => fastembed::EmbeddingModel::BGEBaseENV15,
        "BAAI/bge-large-en-v1.5" | "bge-large-en-v1.5" => fastembed::EmbeddingModel::BGELargeENV15,
        "sentence-transformers/all-MiniLM-L6-v2" | "all-MiniLM-L6-v2" => fastembed::EmbeddingModel::AllMiniLML6V2,
        _ => {
            eprintln!("[yeti-vectors] Unknown text model '{}', defaulting to BGESmallENV15", name);
            fastembed::EmbeddingModel::BGESmallENV15
        }
    }
}

fn parse_image_model(name: &str) -> fastembed::ImageEmbeddingModel {
    match name {
        "clip-ViT-B-32" | "CLIP-ViT-B-32" | "clip-vit-b-32" => fastembed::ImageEmbeddingModel::ClipVitB32,
        _ => {
            eprintln!("[yeti-vectors] Unknown image model '{}', defaulting to ClipVitB32", name);
            fastembed::ImageEmbeddingModel::ClipVitB32
        }
    }
}

// ============================================================================
// FastEmbedVectorHook — implements VectorHook (sync for dylib safety)
// ============================================================================

pub struct FastEmbedVectorHook;

impl VectorHook for FastEmbedVectorHook {
    fn vectorize_fields(
        &self,
        mut record: serde_json::Value,
        mappings: &[FieldMapping],
    ) -> std::result::Result<serde_json::Value, String> {
        for mapping in mappings {
            let source_value = record.get(&mapping.source);

            // Skip if source field is null or missing
            let Some(source_val) = source_value else {
                continue;
            };
            if source_val.is_null() {
                continue;
            }

            let embedding = match mapping.field_type.as_str() {
                "image" => {
                    // Image: decode base64 → raw bytes → ImageEmbedding
                    let base64_str = source_val.as_str()
                        .ok_or_else(|| format!("Image field '{}' must be a base64 string", mapping.source))?;

                    let bytes = base64::Engine::decode(
                        &base64::engine::general_purpose::STANDARD,
                        base64_str,
                    ).map_err(|e| format!("Failed to decode base64 from '{}': {}", mapping.source, e))?;

                    self.vectorize_image(&bytes, &mapping.model)?
                }
                _ => {
                    // Text (default): read string → TextEmbedding
                    let text = source_val.as_str()
                        .ok_or_else(|| format!("Text field '{}' must be a string", mapping.source))?;

                    if text.is_empty() {
                        continue; // Skip empty strings
                    }

                    self.vectorize_text(text, &mapping.model)?
                }
            };

            // Write embedding vector to target field
            let embedding_json: Vec<serde_json::Value> = embedding
                .into_iter()
                .map(|f| serde_json::Value::from(f))
                .collect();

            if let Some(obj) = record.as_object_mut() {
                obj.insert(mapping.target.clone(), serde_json::Value::Array(embedding_json));
            }
        }

        Ok(record)
    }

    fn vectorize_text(
        &self,
        text: &str,
        model: &str,
    ) -> std::result::Result<Vec<f32>, String> {
        let model_arc = get_or_init_text_model(model)?;
        let model_guard = model_arc.lock()
            .map_err(|e| format!("Text model mutex poisoned: {}", e))?;

        let embeddings = model_guard.embed(vec![text.to_string()], None)
            .map_err(|e| format!("Text embedding failed: {}", e))?;

        embeddings.into_iter().next()
            .ok_or_else(|| "Text embedding returned empty result".to_string())
    }

    fn vectorize_fields_batch(
        &self,
        mut records: Vec<serde_json::Value>,
        mappings: &[FieldMapping],
    ) -> std::result::Result<Vec<serde_json::Value>, String> {
        // For each text mapping, collect all texts, embed in one batch, assign back
        for mapping in mappings {
            if mapping.field_type != "text" && !mapping.field_type.is_empty() {
                // Image fields: fall back to per-record
                for record in &mut records {
                    if let Some(src) = record.get(&mapping.source).and_then(|v| v.as_str()) {
                        if !src.is_empty() {
                            if let Ok(embedding) = self.vectorize_text(src, &mapping.model) {
                                let vec_json: Vec<serde_json::Value> = embedding.into_iter().map(serde_json::Value::from).collect();
                                if let Some(obj) = record.as_object_mut() {
                                    obj.insert(mapping.target.clone(), serde_json::Value::Array(vec_json));
                                }
                            }
                        }
                    }
                }
                continue;
            }

            // Collect (index, text) pairs for records that have the source field
            let mut texts: Vec<(usize, String)> = Vec::with_capacity(records.len());
            for (i, record) in records.iter().enumerate() {
                if let Some(val) = record.get(&mapping.source) {
                    if let Some(text) = val.as_str() {
                        if !text.is_empty() {
                            texts.push((i, text.to_string()));
                        }
                    }
                }
            }

            if texts.is_empty() {
                continue;
            }

            // Batch embed all texts in one call
            let model_arc = get_or_init_text_model(&mapping.model)?;
            let model_guard = model_arc.lock()
                .map_err(|e| format!("Text model mutex poisoned: {}", e))?;

            let text_strings: Vec<String> = texts.iter().map(|(_, t)| t.clone()).collect();
            let embeddings = model_guard.embed(text_strings, None)
                .map_err(|e| format!("Batch text embedding failed: {}", e))?;

            // Assign embeddings back to records
            for ((idx, _), embedding) in texts.iter().zip(embeddings.into_iter()) {
                let vec_json: Vec<serde_json::Value> = embedding.into_iter().map(serde_json::Value::from).collect();
                if let Some(obj) = records[*idx].as_object_mut() {
                    obj.insert(mapping.target.clone(), serde_json::Value::Array(vec_json));
                }
            }
        }

        Ok(records)
    }

    fn vectorize_image(
        &self,
        bytes: &[u8],
        model: &str,
    ) -> std::result::Result<Vec<f32>, String> {
        let model_arc = get_or_init_image_model(model)?;
        let model_guard = model_arc.lock()
            .map_err(|e| format!("Image model mutex poisoned: {}", e))?;

        let embeddings = model_guard.embed_bytes(&[bytes], None)
            .map_err(|e| format!("Image embedding failed: {}", e))?;

        embeddings.into_iter().next()
            .ok_or_else(|| "Image embedding returned empty result".to_string())
    }
}

// ============================================================================
// VectorsExtension — implements Extension trait
// ============================================================================

#[derive(Default)]
pub struct VectorsExtension;

impl Extension for VectorsExtension {
    fn name(&self) -> &str {
        "vectors"
    }

    fn vector_hooks(&self) -> Vec<Arc<dyn VectorHook>> {
        vec![Arc::new(FastEmbedVectorHook)]
    }

    fn on_ready(&self, ctx: &ExtensionContext) -> yeti_core::error::Result<()> {
        let dir = PathBuf::from(ctx.root_dir()).join("models");
        eprintln!("[yeti-vectors] Model cache directory: {}", dir.display());
        let _ = models_dir().set(dir);
        Ok(())
    }
}

// Stub resource required by the compiler (derives type name from filename)
resource!(Vectors {
    get => json!({"extension": "yeti-vectors", "status": "active"})
});
