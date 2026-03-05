#ifndef MLX_LLM_H
#define MLX_LLM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct llm_model_ llm_model_t;

/* Load a quantized LLaMA model from a safetensors file.
 * Returns NULL on failure. */
llm_model_t* llm_load(
    const char* safetensors_path,
    int num_layers,
    int hidden_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int intermediate_size,
    int vocab_size,
    float rms_norm_eps,
    float rope_theta,
    int rope_dims);

/* Run a causal forward pass over the full token sequence.
 * token_ids: array of n_tokens token ids (int32)
 * out_logits: caller-allocated float[vocab_size] buffer; filled with
 *             the logits for the next token after the last input token.
 * Returns 0 on success, non-zero on error. */
int llm_forward(
    llm_model_t* model,
    const int32_t* token_ids,
    int n_tokens,
    float* out_logits,
    int vocab_size);

/* Free a model. */
void llm_free(llm_model_t* model);

/* ── KV-cache API (O(n) decode) ──────────────────────────────────────────── */

typedef struct llm_cache_ llm_cache_t;

/* Prefill: run a full prompt through the model, populate the KV cache, and
 * write logits for the next token into out_logits[0..vocab_size-1].
 * Returns a heap-allocated cache on success, NULL on failure. */
llm_cache_t* llm_prefill(
    llm_model_t* model,
    const int32_t* token_ids,
    int n_tokens,
    float* out_logits,
    int vocab_size);

/* Decode: run a single new token using the existing KV cache.
 * Updates the cache in-place (appends one position per layer).
 * Writes logits for the next token into out_logits[0..vocab_size-1].
 * Returns 0 on success, non-zero on failure. */
int llm_decode(
    llm_model_t* model,
    llm_cache_t* cache,
    int32_t token_id,
    float* out_logits,
    int vocab_size);

/* Decode step (async phase 1): build the graph + start async GPU eval.
 * Returns immediately. Call llm_decode_read() to get the results. */
int llm_decode_step(
    llm_model_t* model,
    llm_cache_t* cache,
    int32_t token_id);

/* Decode read (async phase 2): wait for GPU results from previous
 * decode_step, copy logits into out_logits.  Blocks until done. */
int llm_decode_read(
    llm_model_t* model,
    llm_cache_t* cache,
    float* out_logits,
    int vocab_size);

/* Double-buffered generation with GPU-side sampling.
 * Prefill must be called first. Generates up to max_tokens tokens starting
 * from first_token_id (the first sampled token after prefill).
 * on_token callback receives each generated token; return non-zero to stop.
 * Returns 0 on success, non-zero on error. */
int llm_pipeline_generate(
    llm_model_t* model,
    llm_cache_t* cache,
    int32_t      first_token_id,
    int          max_tokens,
    int          eos_id,
    float        temperature,
    int        (*on_token)(int32_t token_id, void* ctx),
    void*        ctx);

/* Free a KV cache and all its GPU tensors. */
void llm_cache_free(llm_cache_t* cache);

#ifdef __cplusplus
}
#endif

#endif /* MLX_LLM_H */
