/* mlx_llm_cpp.cpp — C++ native decode loop for maximum throughput.
 *
 * Implements transformer_layer_decode and the double-buffered generation
 * pipeline using MLX C++ API directly, eliminating ~1155 heap-allocated
 * mlx_array handle wrapper operations per token.
 *
 * All public functions are extern "C" so they link seamlessly with the
 * existing C code and Scala Native FFI.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <optional>

#include "mlx/mlx.h"
#include "mlx/c/mlx.h"
#include "mlx/c/private/array.h"
#include "mlx/c/private/stream.h"
#include "mlx/c/private/vector.h"

namespace mx = mlx::core;

/* ─── Weight references for one transformer layer ─────────────────── */

struct LayerWeights {
    const mx::array& q_w;  const mx::array& q_sc;  const mx::array& q_bi;
    const mx::array& k_w;  const mx::array& k_sc;  const mx::array& k_bi;
    const mx::array& v_w;  const mx::array& v_sc;  const mx::array& v_bi;
    const mx::array& o_w;  const mx::array& o_sc;  const mx::array& o_bi;
    const mx::array& input_norm_w;
    const mx::array* q_norm_w;   /* nullptr if absent */
    const mx::array* k_norm_w;   /* nullptr if absent */
    const mx::array& post_attn_norm_w;
    const mx::array& gate_w; const mx::array& gate_sc; const mx::array& gate_bi;
    const mx::array& up_w;   const mx::array& up_sc;   const mx::array& up_bi;
    const mx::array& down_w; const mx::array& down_sc; const mx::array& down_bi;
};

/* ─── Model context: native C++ references ────────────────────────── */

struct ModelCtx {
    mx::Stream stream;
    int num_layers, hidden_size, num_heads, num_kv_heads, head_dim;
    int intermediate_size, vocab_size;
    float rms_norm_eps, rope_theta;
    int rope_dims;
    int lm_head_mode;

    /* Embedding */
    const mx::array& embed_w;
    const mx::array* embed_sc;  /* nullptr if float */
    const mx::array* embed_bi;  /* nullptr if float or no bias */

    /* Final norm + LM head */
    const mx::array& final_norm_w;
    const mx::array& lm_head_w;
    const mx::array* lm_head_sc;
    const mx::array* lm_head_bi;

    /* Per-layer weights */
    std::vector<LayerWeights> layers;
};

/* ─── Inline helpers ──────────────────────────────────────────────── */

static inline mx::array qmatmul(const mx::array& x,
                                  const mx::array& w,
                                  const mx::array& sc,
                                  const mx::array& bi,
                                  mx::StreamOrDevice s) {
    return mx::quantized_matmul(x, w, sc, bi, true, 64, 4, "affine", s);
}

static inline mx::array silu(const mx::array& x, mx::StreamOrDevice s) {
    return mx::multiply(mx::sigmoid(x, s), x, s);
}

/* Compiled SwiGLU function for mx::compile */
static std::vector<mx::array> swiglu_fn(const std::vector<mx::array>& inputs) {
    auto s = mx::default_stream(mx::default_device());
    auto gate = inputs[0];
    auto up = inputs[1];
    return {mx::multiply(silu(gate, s), up, s)};
}

/* Global compiled closure — initialized once on first use */
static std::function<std::vector<mx::array>(const std::vector<mx::array>&)>
    g_swiglu_compiled;
static bool g_swiglu_initialized = false;

static void init_compiled_swiglu() {
    if (!g_swiglu_initialized) {
        g_swiglu_compiled = mx::compile(swiglu_fn, true);
        g_swiglu_initialized = true;
    }
}

/* ─── Embedding lookup from scalar int32 array (can be lazy) ──────── */

static mx::array embed_lookup_array(const mx::array& token_arr,
                                     const ModelCtx& m) {
    auto s = m.stream;
    auto indices = mx::reshape(token_arr, {1}, s);
    auto w_idx = mx::take(m.embed_w, indices, 0, s);

    if (m.embed_sc) {
        auto sc_idx = mx::take(*m.embed_sc, indices, 0, s);
        std::optional<mx::array> bi_idx = std::nullopt;
        if (m.embed_bi)
            bi_idx = mx::take(*m.embed_bi, indices, 0, s);
        return mx::dequantize(w_idx, sc_idx, bi_idx, 64, 4, "affine",
                              std::nullopt, s);
    }
    return w_idx;
}

/* ─── Single transformer layer decode (pure C++) ──────────────────── */

static mx::array transformer_layer_decode_cpp(
    const mx::array& x,
    mx::array& k_cache,   /* [KH, max_tokens, D] — mutated */
    mx::array& v_cache,   /* [KH, max_tokens, D] — mutated */
    int n_cached,
    const LayerWeights& lw,
    const ModelCtx& m)
{
    auto s = m.stream;
    int H  = m.num_heads;
    int KH = m.num_kv_heads;
    int D  = m.head_dim;

    /* Input norm */
    auto h = mx::fast::rms_norm(x, lw.input_norm_w, m.rms_norm_eps, s);

    /* Q/K/V projections */
    auto Q = qmatmul(h, lw.q_w, lw.q_sc, lw.q_bi, s);
    auto K = qmatmul(h, lw.k_w, lw.k_sc, lw.k_bi, s);
    auto V = qmatmul(h, lw.v_w, lw.v_sc, lw.v_bi, s);

    /* Reshape [1, H*D] → [1, H, D] then transpose → [H, 1, D] */
    Q = mx::transpose(mx::reshape(Q, {1, H,  D}, s), {1, 0, 2}, s);
    K = mx::transpose(mx::reshape(K, {1, KH, D}, s), {1, 0, 2}, s);
    V = mx::transpose(mx::reshape(V, {1, KH, D}, s), {1, 0, 2}, s);

    /* Optional QK-norm */
    if (lw.q_norm_w)
        Q = mx::fast::rms_norm(Q, *lw.q_norm_w, m.rms_norm_eps, s);
    if (lw.k_norm_w)
        K = mx::fast::rms_norm(K, *lw.k_norm_w, m.rms_norm_eps, s);

    /* RoPE: need 4D [B, H, T, D] — expand dim 0, apply, squeeze back */
    std::optional<float> rope_base{m.rope_theta};
    Q = mx::squeeze(mx::fast::rope(mx::expand_dims(Q, 0, s),
                                    m.rope_dims, false, rope_base,
                                    1.0f, n_cached, std::nullopt, s),
                    0, s);
    K = mx::squeeze(mx::fast::rope(mx::expand_dims(K, 0, s),
                                    m.rope_dims, false, rope_base,
                                    1.0f, n_cached, std::nullopt, s),
                    0, s);

    /* Write K/V into cache at position n_cached */
    k_cache = mx::slice_update(k_cache, K,
        {0, n_cached, 0}, {KH, n_cached + 1, D}, {1, 1, 1}, s);
    v_cache = mx::slice_update(v_cache, V,
        {0, n_cached, 0}, {KH, n_cached + 1, D}, {1, 1, 1}, s);

    /* Slice active K/V for SDPA: [:, :n_cached+1, :] */
    auto K_active = mx::slice(k_cache, {0, 0, 0}, {KH, n_cached + 1, D}, s);
    auto V_active = mx::slice(v_cache, {0, 0, 0}, {KH, n_cached + 1, D}, s);

    /* Expand batch dim for SDPA: [H, 1, D] → [1, H, 1, D] */
    auto Qb = mx::expand_dims(Q, 0, s);
    auto Kb = mx::expand_dims(K_active, 0, s);
    auto Vb = mx::expand_dims(V_active, 0, s);

    /* SDPA */
    float scale = 1.0f / std::sqrt(static_cast<float>(D));
    auto attn = mx::fast::scaled_dot_product_attention(
        Qb, Kb, Vb, scale, "", std::nullopt, std::nullopt, s);

    /* Remove batch → [H,1,D]; transpose → [1,H,D]; reshape → [1, H*D] */
    attn = mx::reshape(
        mx::transpose(mx::squeeze(attn, 0, s), {1, 0, 2}, s),
        {1, H * D}, s);

    /* Output projection + residual */
    auto attn_out = qmatmul(attn, lw.o_w, lw.o_sc, lw.o_bi, s);
    auto x1 = mx::add(x, attn_out, s);

    /* Post-attention norm → MLP */
    auto h2 = mx::fast::rms_norm(x1, lw.post_attn_norm_w, m.rms_norm_eps, s);

    auto gate = qmatmul(h2, lw.gate_w, lw.gate_sc, lw.gate_bi, s);
    auto up   = qmatmul(h2, lw.up_w,   lw.up_sc,   lw.up_bi, s);

    /* SwiGLU via compiled closure: fuses sigmoid+mul+mul into 1 GPU kernel */
    auto ffn = g_swiglu_compiled({gate, up})[0];

    /* Down projection + residual */
    auto down = qmatmul(ffn, lw.down_w, lw.down_sc, lw.down_bi, s);
    return mx::add(x1, down, s);
}

/* ─── LM head: x [1, hidden] → logits32 [1, vocab_size] ──────────── */

static mx::array lm_head_graph(const mx::array& x, const ModelCtx& m) {
    auto s = m.stream;
    auto logits = (m.lm_head_mode == 2 || m.lm_head_mode == 3)
        ? qmatmul(x, m.lm_head_w, *m.lm_head_sc, *m.lm_head_bi, s)
        : mx::matmul(x, mx::transpose(m.lm_head_w, s), s);
    return mx::astype(logits, mx::float32, s);
}

/* ─── Decode + sample (pure C++) ──────────────────────────────────── */

static mx::array decode_and_sample_cpp(
    ModelCtx& ctx,
    std::vector<mx::array>& k_caches,
    std::vector<mx::array>& v_caches,
    int& n_cached,
    const mx::array& token_arr,
    float temperature)
{
    auto s = ctx.stream;

    /* Embedding */
    auto x = embed_lookup_array(token_arr, ctx);

    /* Transformer layers */
    for (int i = 0; i < ctx.num_layers; i++) {
        x = transformer_layer_decode_cpp(
            x, k_caches[i], v_caches[i], n_cached, ctx.layers[i], ctx);
    }
    n_cached++;

    /* Final norm → LM head */
    x = mx::fast::rms_norm(x, ctx.final_norm_w, ctx.rms_norm_eps, s);
    auto logits32 = lm_head_graph(x, ctx);

    /* GPU-side sampling */
    auto sampled = (temperature <= 0.0f)
        ? mx::argmax(logits32, -1, false, s)
        : mx::astype(
              mx::random::categorical(
                  mx::divide(logits32, mx::array(temperature), s), -1,
                  std::nullopt, s),
              mx::int32, s);

    /* Async eval: ONLY the sampled token.
     * KV caches are dependencies of sampled → transitively evaluated.
     * Matching Python mlx-lm: mx.async_eval(y, logprobs) — NOT caches.
     * Passing 56 extra cache arrays caused unnecessary overhead. */
    mx::async_eval(sampled);

    return sampled;
}

/* ─── Build ModelCtx from opaque C handles ────────────────────────── */
/* We need to access the internals of llm_model_ and llm_cache_.
 * Since those are defined in mlx_llm.c (C), we replicate the struct
 * layout here. This is safe because both are compiled in the same
 * translation unit group and linked together. */

/* Must match llm_layer_weights_ in mlx_llm.c */
struct llm_layer_weights_c {
    mlx_array q_w, q_sc, q_bi;
    mlx_array k_w, k_sc, k_bi;
    mlx_array v_w, v_sc, v_bi;
    mlx_array o_w, o_sc, o_bi;
    mlx_array input_norm_w;
    mlx_array q_norm_w;
    mlx_array k_norm_w;
    mlx_array post_attn_norm_w;
    mlx_array gate_w, gate_sc, gate_bi;
    mlx_array up_w, up_sc, up_bi;
    mlx_array down_w, down_sc, down_bi;
};

/* Must match llm_model_ in mlx_llm.c */
struct llm_model_c {
    mlx_map_string_to_array  weights;
    mlx_map_string_to_string metadata;
    mlx_stream stream;
    int num_layers;
    int hidden_size;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int intermediate_size;
    int vocab_size;
    float rms_norm_eps;
    float rope_theta;
    int rope_dims;
    int lm_head_mode;
    llm_layer_weights_c* layer_w;
    mlx_array embed_w, embed_sc, embed_bi;
    mlx_array final_norm_w;
    mlx_array lm_head_w, lm_head_sc, lm_head_bi;
    mlx_closure swiglu_compiled;
};

/* Must match llm_cache_ in mlx_llm.c */
struct llm_cache_c {
    int n_layers;
    int n_tokens;
    int max_tokens;
    mlx_array* k;
    mlx_array* v;
    mlx_array pending_logits;
    mlx_vector_array eval_batch;
};

/* Helper: get native ref from mlx_array, or nullptr if null handle */
static inline const mx::array* arr_ptr(mlx_array a) {
    return a.ctx ? static_cast<mx::array*>(a.ctx) : nullptr;
}
static inline mx::array& arr_ref(mlx_array a) {
    return *static_cast<mx::array*>(a.ctx);
}

/* ─── Public extern "C" entry point ───────────────────────────────── */

extern "C" int llm_pipeline_generate_cpp(
    void*    model_ptr,   /* llm_model_t* cast to void* */
    void*    cache_ptr,   /* llm_cache_t* cast to void* */
    int32_t  first_token_id,
    int      max_tokens,
    int      eos_id,
    float    temperature,
    int    (*on_token)(int32_t token_id, void* ctx),
    void*    ctx)
{
    auto* mc = static_cast<llm_model_c*>(model_ptr);
    auto* cc = static_cast<llm_cache_c*>(cache_ptr);

    /* Build ModelCtx — extract native C++ references from C handles */
    ModelCtx mctx{
        .stream       = mlx_stream_get_(mc->stream),
        .num_layers   = mc->num_layers,
        .hidden_size  = mc->hidden_size,
        .num_heads    = mc->num_heads,
        .num_kv_heads = mc->num_kv_heads,
        .head_dim     = mc->head_dim,
        .intermediate_size = mc->intermediate_size,
        .vocab_size   = mc->vocab_size,
        .rms_norm_eps = mc->rms_norm_eps,
        .rope_theta   = mc->rope_theta,
        .rope_dims    = mc->rope_dims,
        .lm_head_mode = mc->lm_head_mode,
        .embed_w      = arr_ref(mc->embed_w),
        .embed_sc     = arr_ptr(mc->embed_sc),
        .embed_bi     = arr_ptr(mc->embed_bi),
        .final_norm_w = arr_ref(mc->final_norm_w),
        .lm_head_w    = arr_ref(mc->lm_head_w),
        .lm_head_sc   = arr_ptr(mc->lm_head_sc),
        .lm_head_bi   = arr_ptr(mc->lm_head_bi),
        .layers       = {},
    };

    /* Build per-layer weight references */
    mctx.layers.reserve(mc->num_layers);
    for (int i = 0; i < mc->num_layers; i++) {
        auto& lw = mc->layer_w[i];
        mctx.layers.push_back(LayerWeights{
            .q_w = arr_ref(lw.q_w), .q_sc = arr_ref(lw.q_sc), .q_bi = arr_ref(lw.q_bi),
            .k_w = arr_ref(lw.k_w), .k_sc = arr_ref(lw.k_sc), .k_bi = arr_ref(lw.k_bi),
            .v_w = arr_ref(lw.v_w), .v_sc = arr_ref(lw.v_sc), .v_bi = arr_ref(lw.v_bi),
            .o_w = arr_ref(lw.o_w), .o_sc = arr_ref(lw.o_sc), .o_bi = arr_ref(lw.o_bi),
            .input_norm_w = arr_ref(lw.input_norm_w),
            .q_norm_w = arr_ptr(lw.q_norm_w),
            .k_norm_w = arr_ptr(lw.k_norm_w),
            .post_attn_norm_w = arr_ref(lw.post_attn_norm_w),
            .gate_w = arr_ref(lw.gate_w), .gate_sc = arr_ref(lw.gate_sc), .gate_bi = arr_ref(lw.gate_bi),
            .up_w = arr_ref(lw.up_w),     .up_sc = arr_ref(lw.up_sc),     .up_bi = arr_ref(lw.up_bi),
            .down_w = arr_ref(lw.down_w), .down_sc = arr_ref(lw.down_sc), .down_bi = arr_ref(lw.down_bi),
        });
    }

    /* Extract K/V cache arrays as native C++ arrays.
     * We'll work with these directly, then write them back at the end. */
    int n_cached = cc->n_tokens;
    std::vector<mx::array> k_caches;
    std::vector<mx::array> v_caches;
    k_caches.reserve(mc->num_layers);
    v_caches.reserve(mc->num_layers);
    for (int i = 0; i < mc->num_layers; i++) {
        k_caches.push_back(arr_ref(cc->k[i]));
        v_caches.push_back(arr_ref(cc->v[i]));
    }

    try {
        init_compiled_swiglu();
        /* Initial token as evaluated scalar array */
        auto y = mx::array(first_token_id);

        /* Build first decode graph */
        auto sampled = decode_and_sample_cpp(
            mctx, k_caches, v_caches, n_cached, y, temperature);

        for (int n = 0; n < max_tokens; n++) {
            /* Build NEXT graph using current sampled (still lazy!) */
            auto next_sampled = decode_and_sample_cpp(
                mctx, k_caches, v_caches, n_cached, sampled, temperature);

            /* Block on CURRENT result — .item() calls eval() internally */
            int32_t tok = sampled.item<int32_t>();

            /* Deliver token */
            if (on_token && on_token(tok, ctx) != 0) {
                /* Write back caches before returning */
                goto write_back;
            }

            if (tok == eos_id) {
                goto write_back;
            }

            sampled = std::move(next_sampled);
        }

write_back:
        /* Write updated K/V caches back to C handles */
        for (int i = 0; i < mc->num_layers; i++) {
            mlx_array_set_(cc->k[i], std::move(k_caches[i]));
            mlx_array_set_(cc->v[i], std::move(v_caches[i]));
        }
        cc->n_tokens = n_cached;
        return 0;

    } catch (const std::exception& e) {
        fprintf(stderr, "mlx_llm_cpp error: %s\n", e.what());
        /* Still try to write back caches */
        for (int i = 0; i < mc->num_layers; i++) {
            mlx_array_set_(cc->k[i], std::move(k_caches[i]));
            mlx_array_set_(cc->v[i], std::move(v_caches[i]));
        }
        cc->n_tokens = n_cached;
        return -1;
    }
}
