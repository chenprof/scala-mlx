#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mlx/c/mlx.h"
#include "mlx_llm.h"

/* ------------------------------------------------------------------
 * Model struct
 * ------------------------------------------------------------------ */

/* Per-layer cached weight pointers — resolved once at load time,
 * eliminating 784 string-based hash lookups per decode step. */
typedef struct llm_layer_weights_ {
    /* Attention projections (quantized: weight/scales/biases) */
    mlx_array q_w, q_sc, q_bi;
    mlx_array k_w, k_sc, k_bi;
    mlx_array v_w, v_sc, v_bi;
    mlx_array o_w, o_sc, o_bi;
    /* Norms */
    mlx_array input_norm_w;
    mlx_array q_norm_w;        /* may be null (optional QK-norm) */
    mlx_array k_norm_w;        /* may be null */
    mlx_array post_attn_norm_w;
    /* MLP projections */
    mlx_array gate_w, gate_sc, gate_bi;
    mlx_array up_w, up_sc, up_bi;
    mlx_array down_w, down_sc, down_bi;
} llm_layer_weights_t;

struct llm_model_ {
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
    /* 0 = weight-tied float   (embed_tokens.weight, float matmul)
     * 1 = separate float      (lm_head.weight, float matmul)
     * 2 = separate quantized  (lm_head.weight/scales/biases)
     * 3 = weight-tied quant   (embed_tokens.weight/scales/biases) */
    int lm_head_mode;
    /* Cached weight pointers (resolved once in llm_load) */
    llm_layer_weights_t* layer_w;   /* [num_layers] */
    mlx_array embed_w, embed_sc, embed_bi;
    mlx_array final_norm_w;
    mlx_array lm_head_w, lm_head_sc, lm_head_bi;
    /* Compiled SwiGLU closure: [gate, up] → [silu(gate) * up] */
    mlx_closure swiglu_compiled;
};

/* ------------------------------------------------------------------
 * Internal helpers
 * ------------------------------------------------------------------ */

static mlx_array get_w(const llm_model_t* m, const char* key) {
    mlx_array arr = mlx_array_new();
    mlx_map_string_to_array_get(&arr, m->weights, key);
    return arr;
}

static int is_null_arr(mlx_array a) { return a.ctx == NULL; }

/* SiLU(x) = x * sigmoid(x) */
static int do_silu(mlx_array* res, mlx_array x, mlx_stream s) {
    mlx_array sig = mlx_array_new();
    if (mlx_sigmoid(&sig, x, s) != 0) return -1;
    int err = mlx_multiply(res, x, sig, s);
    mlx_array_free(sig);
    return err;
}

/* SwiGLU closure for mlx_compile: inputs=[gate, up], returns=[silu(gate)*up].
 * Fuses sigmoid + multiply + multiply into one compiled kernel. */
static int swiglu_closure_fn(mlx_vector_array* res,
                              const mlx_vector_array inputs) {
    mlx_stream s = mlx_default_gpu_stream_new();
    mlx_array gate = mlx_array_new();
    mlx_array up   = mlx_array_new();
    mlx_vector_array_get(&gate, inputs, 0);
    mlx_vector_array_get(&up,   inputs, 1);

    mlx_array sg = mlx_array_new();
    if (do_silu(&sg, gate, s) != 0) {
        mlx_array_free(gate); mlx_array_free(up); mlx_stream_free(s);
        return 1;
    }
    mlx_array_free(gate);

    mlx_array ffn = mlx_array_new();
    mlx_multiply(&ffn, sg, up, s);
    mlx_array_free(sg); mlx_array_free(up);

    mlx_vector_array_append_value(*res, ffn);
    mlx_array_free(ffn);
    mlx_stream_free(s);
    return 0;
}

/* Quantized matmul: result = x @ w.T  (int4, group_size=64) */
static int do_qmatmul(mlx_array* res, mlx_array x,
                      const char* pfx, const llm_model_t* m) {
    char wk[512], sk[512], bk[512];
    snprintf(wk, sizeof(wk), "%s.weight", pfx);
    snprintf(sk, sizeof(sk), "%s.scales", pfx);
    snprintf(bk, sizeof(bk), "%s.biases", pfx);

    mlx_array w  = get_w(m, wk);
    mlx_array sc = get_w(m, sk);
    mlx_array bi = get_w(m, bk);

    mlx_optional_int gs = { .value = 64, .has_value = true };
    mlx_optional_int bt = { .value = 4,  .has_value = true };

    int err = mlx_quantized_matmul(
        res, x, w, sc, bi,
        /*transpose=*/true, gs, bt, "affine", m->stream);

    mlx_array_free(w);
    mlx_array_free(sc);
    mlx_array_free(bi);
    return err;
}

/* Float matmul: result = x @ w.T */
static int do_fmatmul_t(mlx_array* res, mlx_array x,
                        mlx_array w, mlx_stream s) {
    mlx_array wt = mlx_array_new();
    if (mlx_transpose(&wt, w, s) != 0) return -1;
    int err = mlx_matmul(res, x, wt, s);
    mlx_array_free(wt);
    return err;
}

/* Embedding lookup: handles both quantized and float weights.
 * Uses cached m->embed_w/sc/bi pointers. */
static int do_embed_lookup(mlx_array* out, const int32_t* token_ids,
                           int n_tokens, const llm_model_t* m) {
    mlx_stream s = m->stream;
    mlx_array indices = mlx_array_new_data(
        (const void*)token_ids, &n_tokens, 1, MLX_INT32);

    if (is_null_arr(m->embed_w)) {
        fprintf(stderr, "mlx_llm: missing model.embed_tokens.weight\n");
        mlx_array_free(indices);
        return -1;
    }

    mlx_array w_idx = mlx_array_new();
    mlx_take_axis(&w_idx, m->embed_w, indices, 0, s);

    if (!is_null_arr(m->embed_sc)) {
        mlx_array sc_idx = mlx_array_new();
        mlx_take_axis(&sc_idx, m->embed_sc, indices, 0, s);

        mlx_array bi_idx = mlx_array_new();
        if (!is_null_arr(m->embed_bi))
            mlx_take_axis(&bi_idx, m->embed_bi, indices, 0, s);
        mlx_array_free(indices);

        mlx_optional_int gs   = { .value = 64, .has_value = true };
        mlx_optional_int bits = { .value = 4,  .has_value = true };
        mlx_optional_dtype dt = { .has_value = false };
        int err = mlx_dequantize(out, w_idx, sc_idx, bi_idx, gs, bits, "affine", dt, s);
        mlx_array_free(w_idx); mlx_array_free(sc_idx); mlx_array_free(bi_idx);
        return err;
    } else {
        mlx_array_free(indices);
        *out = w_idx;
        return 0;
    }
}

/* Embedding lookup from an mlx_array scalar int32 token (can be unevaluated/lazy).
 * Reshapes scalar → [1] for mlx_take_axis, then embeds → [1, hidden].
 * This is the foundation for double-buffered pipelining: when token_arr is
 * unevaluated, the entire graph remains lazy. */
static int do_embed_lookup_array(mlx_array* out, mlx_array token_arr,
                                  const llm_model_t* m) {
    mlx_stream s = m->stream;

    /* Reshape scalar → [1] */
    int sh1[1] = {1};
    mlx_array indices = mlx_array_new();
    mlx_reshape(&indices, token_arr, sh1, 1, s);

    if (is_null_arr(m->embed_w)) {
        fprintf(stderr, "mlx_llm: missing model.embed_tokens.weight\n");
        mlx_array_free(indices);
        return -1;
    }

    mlx_array w_idx = mlx_array_new();
    mlx_take_axis(&w_idx, m->embed_w, indices, 0, s);

    if (!is_null_arr(m->embed_sc)) {
        mlx_array sc_idx = mlx_array_new();
        mlx_take_axis(&sc_idx, m->embed_sc, indices, 0, s);

        mlx_array bi_idx = mlx_array_new();
        if (!is_null_arr(m->embed_bi))
            mlx_take_axis(&bi_idx, m->embed_bi, indices, 0, s);
        mlx_array_free(indices);

        mlx_optional_int gs   = { .value = 64, .has_value = true };
        mlx_optional_int bits = { .value = 4,  .has_value = true };
        mlx_optional_dtype dt = { .has_value = false };
        int err = mlx_dequantize(out, w_idx, sc_idx, bi_idx, gs, bits, "affine", dt, s);
        mlx_array_free(w_idx); mlx_array_free(sc_idx); mlx_array_free(bi_idx);
        return err;
    } else {
        mlx_array_free(indices);
        *out = w_idx;
        return 0;
    }
}

/* RMS norm using weight at "<prefix>.weight" */
static int do_rms_norm(mlx_array* res, mlx_array x,
                       const char* pfx, const llm_model_t* m) {
    char wk[512];
    snprintf(wk, sizeof(wk), "%s.weight", pfx);
    mlx_array w = get_w(m, wk);
    if (is_null_arr(w)) {
        fprintf(stderr, "mlx_llm: missing norm weight: %s\n", wk);
        return -1;
    }
    int err = mlx_fast_rms_norm(res, x, w, m->rms_norm_eps, m->stream);
    mlx_array_free(w);
    return err;
}

/* Cached variants — use pre-resolved weight pointers (no string lookups) */

static int do_qmatmul_cached(mlx_array* res, mlx_array x,
                              mlx_array w, mlx_array sc, mlx_array bi,
                              const llm_model_t* m) {
    mlx_optional_int gs = { .value = 64, .has_value = true };
    mlx_optional_int bt = { .value = 4,  .has_value = true };
    return mlx_quantized_matmul(res, x, w, sc, bi, true, gs, bt, "affine", m->stream);
}

static int do_rms_norm_cached(mlx_array* res, mlx_array x,
                               mlx_array w, const llm_model_t* m) {
    return mlx_fast_rms_norm(res, x, w, m->rms_norm_eps, m->stream);
}

/* ------------------------------------------------------------------
 * Single transformer layer (stateless, full sequence)
 *   x:   [T, hidden]  (full sequence, all positions)
 *   out: [T, hidden]  (residual already added)
 * ------------------------------------------------------------------ */

static int transformer_layer(mlx_array* out_x, mlx_array x,
                             int layer, int T, const llm_model_t* m)
{
    mlx_stream s = m->stream;
    int H  = m->num_heads;
    int KH = m->num_kv_heads;
    int D  = m->head_dim;
    const llm_layer_weights_t* lw = &m->layer_w[layer];

    /* 1. Input norm */
    mlx_array h = mlx_array_new();
    if (do_rms_norm_cached(&h, x, lw->input_norm_w, m) != 0) return -1;

    /* 2. Q / K / V projections */
    mlx_array Q = mlx_array_new();
    mlx_array K = mlx_array_new();
    mlx_array V = mlx_array_new();
    if (do_qmatmul_cached(&Q, h, lw->q_w, lw->q_sc, lw->q_bi, m) != 0 ||
        do_qmatmul_cached(&K, h, lw->k_w, lw->k_sc, lw->k_bi, m) != 0 ||
        do_qmatmul_cached(&V, h, lw->v_w, lw->v_sc, lw->v_bi, m) != 0) {
        mlx_array_free(h);
        mlx_array_free(Q); mlx_array_free(K); mlx_array_free(V);
        return -1;
    }
    mlx_array_free(h);

    /* 3. Reshape: [T, heads*D] → [T, heads, D] */
    { int sh[3]={T,H, D}; mlx_array r=mlx_array_new(); mlx_reshape(&r,Q,sh,3,s); mlx_array_free(Q); Q=r; }
    { int sh[3]={T,KH,D}; mlx_array r=mlx_array_new(); mlx_reshape(&r,K,sh,3,s); mlx_array_free(K); K=r; }
    { int sh[3]={T,KH,D}; mlx_array r=mlx_array_new(); mlx_reshape(&r,V,sh,3,s); mlx_array_free(V); V=r; }

    /* 3b. QK-norm (Qwen3) */
    if (!is_null_arr(lw->q_norm_w)) {
        mlx_array r = mlx_array_new();
        mlx_fast_rms_norm(&r, Q, lw->q_norm_w, m->rms_norm_eps, s);
        mlx_array_free(Q); Q = r;
    }
    if (!is_null_arr(lw->k_norm_w)) {
        mlx_array r = mlx_array_new();
        mlx_fast_rms_norm(&r, K, lw->k_norm_w, m->rms_norm_eps, s);
        mlx_array_free(K); K = r;
    }

    /* 4. Transpose: [T, heads, D] → [heads, T, D] */
    { int p[3]={1,0,2}; mlx_array r=mlx_array_new(); mlx_transpose_axes(&r,Q,p,3,s); mlx_array_free(Q); Q=r; }
    { int p[3]={1,0,2}; mlx_array r=mlx_array_new(); mlx_transpose_axes(&r,K,p,3,s); mlx_array_free(K); K=r; }
    { int p[3]={1,0,2}; mlx_array r=mlx_array_new(); mlx_transpose_axes(&r,V,p,3,s); mlx_array_free(V); V=r; }

    /* 5. RoPE on Q and K */
    mlx_optional_float rope_base = { .value = m->rope_theta, .has_value = true };
    {
        mlx_array nf = mlx_array_new();
        mlx_array r  = mlx_array_new();
        mlx_fast_rope(&r, Q, m->rope_dims, false, rope_base, 1.0f, 0, nf, s);
        mlx_array_free(nf); mlx_array_free(Q); Q = r;
    }
    {
        mlx_array nf = mlx_array_new();
        mlx_array r  = mlx_array_new();
        mlx_fast_rope(&r, K, m->rope_dims, false, rope_base, 1.0f, 0, nf, s);
        mlx_array_free(nf); mlx_array_free(K); K = r;
    }

    /* 6. Add batch dim: [heads,T,D] → [1,heads,T,D] */
    { mlx_array r=mlx_array_new(); mlx_expand_dims(&r,Q,0,s); mlx_array_free(Q); Q=r; }
    { mlx_array r=mlx_array_new(); mlx_expand_dims(&r,K,0,s); mlx_array_free(K); K=r; }
    { mlx_array r=mlx_array_new(); mlx_expand_dims(&r,V,0,s); mlx_array_free(V); V=r; }

    /* 7. Causal scaled dot-product attention */
    float scale = 1.0f / sqrtf((float)D);
    mlx_array attn = mlx_array_new();
    {
        mlx_array nm = mlx_array_new();
        mlx_array ns = mlx_array_new();
        int err = mlx_fast_scaled_dot_product_attention(
            &attn, Q, K, V, scale, "causal", nm, ns, s);
        mlx_array_free(nm); mlx_array_free(ns);
        mlx_array_free(Q); mlx_array_free(K); mlx_array_free(V);
        if (err != 0) return -1;
    }

    /* 8. Remove batch dim → [heads, T, D] */
    { mlx_array r=mlx_array_new(); mlx_squeeze_axis(&r,attn,0,s); mlx_array_free(attn); attn=r; }

    /* 9. Transpose back: [heads,T,D] → [T,heads,D] */
    { int p[3]={1,0,2}; mlx_array r=mlx_array_new(); mlx_transpose_axes(&r,attn,p,3,s); mlx_array_free(attn); attn=r; }

    /* 10. Reshape: [T,heads,D] → [T, H*D] */
    { int sh[2]={T,H*D}; mlx_array r=mlx_array_new(); mlx_reshape(&r,attn,sh,2,s); mlx_array_free(attn); attn=r; }

    /* 11. Output projection */
    mlx_array attn_out = mlx_array_new();
    if (do_qmatmul_cached(&attn_out, attn, lw->o_w, lw->o_sc, lw->o_bi, m) != 0) {
        mlx_array_free(attn); return -1;
    }
    mlx_array_free(attn);

    /* 12. First residual: x = x + attn_out */
    mlx_array x1 = mlx_array_new();
    mlx_add(&x1, x, attn_out, s);
    mlx_array_free(attn_out);

    /* 13. Post-attention norm */
    mlx_array h2 = mlx_array_new();
    if (do_rms_norm_cached(&h2, x1, lw->post_attn_norm_w, m) != 0) { mlx_array_free(x1); return -1; }

    /* 14. SwiGLU FFN */
    mlx_array gate = mlx_array_new();
    mlx_array up   = mlx_array_new();
    if (do_qmatmul_cached(&gate, h2, lw->gate_w, lw->gate_sc, lw->gate_bi, m) != 0 ||
        do_qmatmul_cached(&up,   h2, lw->up_w,   lw->up_sc,   lw->up_bi,   m) != 0) {
        mlx_array_free(h2); mlx_array_free(gate); mlx_array_free(up);
        mlx_array_free(x1); return -1;
    }
    mlx_array_free(h2);

    mlx_array sg = mlx_array_new();
    if (do_silu(&sg, gate, s) != 0) {
        mlx_array_free(gate); mlx_array_free(up); mlx_array_free(x1); return -1;
    }
    mlx_array_free(gate);

    mlx_array ffn = mlx_array_new();
    mlx_multiply(&ffn, sg, up, s);
    mlx_array_free(sg); mlx_array_free(up);

    mlx_array down = mlx_array_new();
    if (do_qmatmul_cached(&down, ffn, lw->down_w, lw->down_sc, lw->down_bi, m) != 0) {
        mlx_array_free(ffn); mlx_array_free(x1); return -1;
    }
    mlx_array_free(ffn);

    /* 15. Second residual: out = x1 + down */
    mlx_add(out_x, x1, down, s);
    mlx_array_free(x1);
    mlx_array_free(down);

    return 0;
}

/* ------------------------------------------------------------------
 * llm_load
 * ------------------------------------------------------------------ */

llm_model_t* llm_load(
    const char* safetensors_path,
    int num_layers, int hidden_size, int num_heads, int num_kv_heads,
    int head_dim, int intermediate_size, int vocab_size,
    float rms_norm_eps, float rope_theta, int rope_dims)
{
    llm_model_t* m = (llm_model_t*)calloc(1, sizeof(llm_model_t));
    if (!m) return NULL;

    m->stream            = mlx_default_gpu_stream_new();
    m->num_layers        = num_layers;
    m->hidden_size       = hidden_size;
    m->num_heads         = num_heads;
    m->num_kv_heads      = num_kv_heads;
    m->head_dim          = head_dim;
    m->intermediate_size = intermediate_size;
    m->vocab_size        = vocab_size;
    m->rms_norm_eps      = rms_norm_eps;
    m->rope_theta        = rope_theta;
    m->rope_dims         = rope_dims;

    m->weights  = mlx_map_string_to_array_new();
    m->metadata = mlx_map_string_to_string_new();

    fprintf(stderr, "mlx_llm: loading %s ...\n", safetensors_path);

    mlx_stream cpu_s = mlx_default_cpu_stream_new();
    int load_err = mlx_load_safetensors(&m->weights, &m->metadata,
                                        safetensors_path, cpu_s);
    mlx_stream_free(cpu_s);
    if (load_err != 0) {
        fprintf(stderr, "mlx_llm: failed to load safetensors\n");
        mlx_stream_free(m->stream);
        free(m);
        return NULL;
    }

    /* Determine lm_head_mode */
    {
        mlx_array lm_w  = get_w(m, "lm_head.weight");
        mlx_array lm_s  = get_w(m, "lm_head.scales");
        mlx_array emb_s = get_w(m, "model.embed_tokens.scales");
        if (!is_null_arr(lm_w) && !is_null_arr(lm_s))
            m->lm_head_mode = 2;
        else if (!is_null_arr(lm_w))
            m->lm_head_mode = 1;
        else if (!is_null_arr(emb_s))
            m->lm_head_mode = 3;
        else
            m->lm_head_mode = 0;
        mlx_array_free(lm_w); mlx_array_free(lm_s); mlx_array_free(emb_s);
    }

    /* ---- Cache weight pointers (resolve all string keys once) ---- */

    /* Global weights */
    m->embed_w  = get_w(m, "model.embed_tokens.weight");
    m->embed_sc = get_w(m, "model.embed_tokens.scales");
    m->embed_bi = get_w(m, "model.embed_tokens.biases");
    m->final_norm_w = get_w(m, "model.norm.weight");

    /* LM head: point to the right weight depending on mode */
    if (m->lm_head_mode == 2) {
        m->lm_head_w  = get_w(m, "lm_head.weight");
        m->lm_head_sc = get_w(m, "lm_head.scales");
        m->lm_head_bi = get_w(m, "lm_head.biases");
    } else if (m->lm_head_mode == 3) {
        /* weight-tied quantized: reuse embed weights */
        m->lm_head_w  = get_w(m, "model.embed_tokens.weight");
        m->lm_head_sc = get_w(m, "model.embed_tokens.scales");
        m->lm_head_bi = get_w(m, "model.embed_tokens.biases");
    } else if (m->lm_head_mode == 1) {
        m->lm_head_w  = get_w(m, "lm_head.weight");
        m->lm_head_sc = mlx_array_new();
        m->lm_head_bi = mlx_array_new();
    } else {
        /* mode 0: weight-tied float */
        m->lm_head_w  = get_w(m, "model.embed_tokens.weight");
        m->lm_head_sc = mlx_array_new();
        m->lm_head_bi = mlx_array_new();
    }

    /* Per-layer weights */
    m->layer_w = (llm_layer_weights_t*)calloc(num_layers, sizeof(llm_layer_weights_t));
    for (int i = 0; i < num_layers; i++) {
        llm_layer_weights_t* lw = &m->layer_w[i];
        char key[512];

#define CACHE_W(field, fmt) \
        snprintf(key, sizeof(key), fmt, i); \
        lw->field = get_w(m, key)

        CACHE_W(q_w,    "model.layers.%d.self_attn.q_proj.weight");
        CACHE_W(q_sc,   "model.layers.%d.self_attn.q_proj.scales");
        CACHE_W(q_bi,   "model.layers.%d.self_attn.q_proj.biases");
        CACHE_W(k_w,    "model.layers.%d.self_attn.k_proj.weight");
        CACHE_W(k_sc,   "model.layers.%d.self_attn.k_proj.scales");
        CACHE_W(k_bi,   "model.layers.%d.self_attn.k_proj.biases");
        CACHE_W(v_w,    "model.layers.%d.self_attn.v_proj.weight");
        CACHE_W(v_sc,   "model.layers.%d.self_attn.v_proj.scales");
        CACHE_W(v_bi,   "model.layers.%d.self_attn.v_proj.biases");
        CACHE_W(o_w,    "model.layers.%d.self_attn.o_proj.weight");
        CACHE_W(o_sc,   "model.layers.%d.self_attn.o_proj.scales");
        CACHE_W(o_bi,   "model.layers.%d.self_attn.o_proj.biases");

        CACHE_W(input_norm_w,    "model.layers.%d.input_layernorm.weight");
        CACHE_W(q_norm_w,        "model.layers.%d.self_attn.q_norm.weight");
        CACHE_W(k_norm_w,        "model.layers.%d.self_attn.k_norm.weight");
        CACHE_W(post_attn_norm_w,"model.layers.%d.post_attention_layernorm.weight");

        CACHE_W(gate_w,  "model.layers.%d.mlp.gate_proj.weight");
        CACHE_W(gate_sc, "model.layers.%d.mlp.gate_proj.scales");
        CACHE_W(gate_bi, "model.layers.%d.mlp.gate_proj.biases");
        CACHE_W(up_w,    "model.layers.%d.mlp.up_proj.weight");
        CACHE_W(up_sc,   "model.layers.%d.mlp.up_proj.scales");
        CACHE_W(up_bi,   "model.layers.%d.mlp.up_proj.biases");
        CACHE_W(down_w,  "model.layers.%d.mlp.down_proj.weight");
        CACHE_W(down_sc, "model.layers.%d.mlp.down_proj.scales");
        CACHE_W(down_bi, "model.layers.%d.mlp.down_proj.biases");

#undef CACHE_W
    }

    /* Compile SwiGLU closure: [gate, up] → [silu(gate) * up]
     * Fuses sigmoid + 2 multiplies into one kernel per layer. */
    {
        mlx_closure raw = mlx_closure_new_func(swiglu_closure_fn);
        m->swiglu_compiled = mlx_closure_new();
        mlx_compile(&m->swiglu_compiled, raw, true);
        mlx_closure_free(raw);
    }

    fprintf(stderr, "mlx_llm: loaded  lm_head_mode=%d  cached %d layers\n",
            m->lm_head_mode, num_layers);
    return m;
}

/* ------------------------------------------------------------------
 * llm_forward — stateless causal forward pass over a full sequence.
 * ------------------------------------------------------------------ */

int llm_forward(
    llm_model_t* m,
    const int32_t* token_ids,
    int n_tokens,
    float* out_logits,
    int vocab_size)
{
    mlx_stream s = m->stream;

    mlx_array x = mlx_array_new();
    if (do_embed_lookup(&x, token_ids, n_tokens, m) != 0) {
        fprintf(stderr, "mlx_llm: embedding lookup failed\n");
        return -1;
    }

    for (int i = 0; i < m->num_layers; i++) {
        mlx_array x_new = mlx_array_new();
        if (transformer_layer(&x_new, x, i, n_tokens, m) != 0) {
            fprintf(stderr, "mlx_llm: layer %d failed\n", i);
            mlx_array_free(x);
            return -1;
        }
        mlx_array_free(x);
        x = x_new;
    }

    {
        mlx_array xn = mlx_array_new();
        if (do_rms_norm_cached(&xn, x, m->final_norm_w, m) != 0) {
            mlx_array_free(x); return -1;
        }
        mlx_array_free(x); x = xn;
    }

    mlx_array last = mlx_array_new();
    {
        int start[2]  = { n_tokens - 1, 0             };
        int stop[2]   = { n_tokens,     m->hidden_size };
        int stride[2] = { 1,            1              };
        mlx_slice(&last, x, start, 2, stop, 2, stride, 2, s);
        mlx_array_free(x);
    }

    mlx_array logits = mlx_array_new();
    if (m->lm_head_mode == 2 || m->lm_head_mode == 3) {
        if (do_qmatmul_cached(&logits, last, m->lm_head_w, m->lm_head_sc, m->lm_head_bi, m) != 0) {
            mlx_array_free(last); return -1;
        }
    } else {
        if (do_fmatmul_t(&logits, last, m->lm_head_w, s) != 0) {
            mlx_array_free(last); return -1;
        }
    }
    mlx_array_free(last);

    mlx_array_eval(logits);
    mlx_synchronize(s);

    mlx_array logits32 = mlx_array_new();
    mlx_astype(&logits32, logits, MLX_FLOAT32, s);
    mlx_array_eval(logits32);
    mlx_synchronize(s);
    mlx_array_free(logits);

    const float* data = mlx_array_data_float32(logits32);
    if (!data) {
        mlx_array_free(logits32); return -1;
    }
    memcpy(out_logits, data, (size_t)vocab_size * sizeof(float));
    mlx_array_free(logits32);
    return 0;
}

/* ------------------------------------------------------------------
 * KV cache struct
 *
 * K/V are stored as pre-allocated [KH, max_tokens, D] buffers.
 * The first n_tokens positions are valid; the rest are zeros.
 * This matches Python mlx-lm's KVCache pattern (slice_update writes).
 * ------------------------------------------------------------------ */

struct llm_cache_ {
    int n_layers;
    int n_tokens;      /* cached sequence length (grows by 1 per decode step) */
    int max_tokens;    /* total allocated capacity of k/v buffers */
    mlx_array* k;      /* k[l]: [KH, max_tokens, D] */
    mlx_array* v;      /* v[l]: [KH, max_tokens, D] */
    mlx_array pending_logits;     /* unevaluated logits from last decode_step */
    mlx_vector_array eval_batch;  /* reusable vector for async_eval calls */
};

/* NOTE: mlx_compile was tested with three approaches and all regressed:
 *   1. Per-layer closures (2×28): 73-86 tok/s — closure call overhead
 *   2. Full-model + mask SDPA:    62-74 tok/s — SDPA on max_tokens positions
 *   3. Full-model + take_axis:    78-81 tok/s — gather slower than slice view
 * Conclusion: mlx_compile helps Python (eliminates interpreter overhead) but
 * not C, where per-call cost is already ~50ns. Uncompiled path is fastest. */

/* ------------------------------------------------------------------
 * transformer_layer_prefill
 *   Same forward pass as transformer_layer, but also outputs K and V
 *   (shape [KH, T, D], RoPE already applied) for the KV cache.
 *   Caller must initialise *out_k / *out_v to mlx_array_new().
 * ------------------------------------------------------------------ */

static int transformer_layer_prefill(
    mlx_array* out_x, mlx_array* out_k, mlx_array* out_v,
    mlx_array x, int layer, int T, const llm_model_t* m)
{
    mlx_stream s = m->stream;
    int H  = m->num_heads;
    int KH = m->num_kv_heads;
    int D  = m->head_dim;
    const llm_layer_weights_t* lw = &m->layer_w[layer];

    mlx_array h = mlx_array_new();
    if (do_rms_norm_cached(&h, x, lw->input_norm_w, m) != 0) return -1;

    mlx_array Q = mlx_array_new();
    mlx_array K = mlx_array_new();
    mlx_array V = mlx_array_new();
    if (do_qmatmul_cached(&Q, h, lw->q_w, lw->q_sc, lw->q_bi, m) != 0 ||
        do_qmatmul_cached(&K, h, lw->k_w, lw->k_sc, lw->k_bi, m) != 0 ||
        do_qmatmul_cached(&V, h, lw->v_w, lw->v_sc, lw->v_bi, m) != 0) {
        mlx_array_free(h);
        mlx_array_free(Q); mlx_array_free(K); mlx_array_free(V);
        return -1;
    }
    mlx_array_free(h);

    { int sh[3]={T,H, D}; mlx_array r=mlx_array_new(); mlx_reshape(&r,Q,sh,3,s); mlx_array_free(Q); Q=r; }
    { int sh[3]={T,KH,D}; mlx_array r=mlx_array_new(); mlx_reshape(&r,K,sh,3,s); mlx_array_free(K); K=r; }
    { int sh[3]={T,KH,D}; mlx_array r=mlx_array_new(); mlx_reshape(&r,V,sh,3,s); mlx_array_free(V); V=r; }

    if (!is_null_arr(lw->q_norm_w)) {
        mlx_array r=mlx_array_new(); mlx_fast_rms_norm(&r,Q,lw->q_norm_w,m->rms_norm_eps,s); mlx_array_free(Q); Q=r; }
    if (!is_null_arr(lw->k_norm_w)) {
        mlx_array r=mlx_array_new(); mlx_fast_rms_norm(&r,K,lw->k_norm_w,m->rms_norm_eps,s); mlx_array_free(K); K=r; }

    { int p[3]={1,0,2}; mlx_array r=mlx_array_new(); mlx_transpose_axes(&r,Q,p,3,s); mlx_array_free(Q); Q=r; }
    { int p[3]={1,0,2}; mlx_array r=mlx_array_new(); mlx_transpose_axes(&r,K,p,3,s); mlx_array_free(K); K=r; }
    { int p[3]={1,0,2}; mlx_array r=mlx_array_new(); mlx_transpose_axes(&r,V,p,3,s); mlx_array_free(V); V=r; }

    /* RoPE — offset 0 for prefill */
    mlx_optional_float rope_base = { .value = m->rope_theta, .has_value = true };
    { mlx_array nf=mlx_array_new(); mlx_array r=mlx_array_new();
      mlx_fast_rope(&r,Q,m->rope_dims,false,rope_base,1.0f,0,nf,s);
      mlx_array_free(nf); mlx_array_free(Q); Q=r; }
    { mlx_array nf=mlx_array_new(); mlx_array r=mlx_array_new();
      mlx_fast_rope(&r,K,m->rope_dims,false,rope_base,1.0f,0,nf,s);
      mlx_array_free(nf); mlx_array_free(K); K=r; }

    /* Save K, V to cache output (shape [KH, T, D], RoPE applied). */
    mlx_array_set(out_k, K);
    mlx_array_set(out_v, V);

    /* Expand batch dim for SDPA */
    { mlx_array r=mlx_array_new(); mlx_expand_dims(&r,Q,0,s); mlx_array_free(Q); Q=r; }
    { mlx_array r=mlx_array_new(); mlx_expand_dims(&r,K,0,s); mlx_array_free(K); K=r; }
    { mlx_array r=mlx_array_new(); mlx_expand_dims(&r,V,0,s); mlx_array_free(V); V=r; }

    float scale = 1.0f / sqrtf((float)D);
    mlx_array attn = mlx_array_new();
    { mlx_array nm=mlx_array_new(); mlx_array ns=mlx_array_new();
      int err = mlx_fast_scaled_dot_product_attention(&attn,Q,K,V,scale,"causal",nm,ns,s);
      mlx_array_free(nm); mlx_array_free(ns);
      mlx_array_free(Q); mlx_array_free(K); mlx_array_free(V);
      if (err != 0) return -1; }

    { mlx_array r=mlx_array_new(); mlx_squeeze_axis(&r,attn,0,s); mlx_array_free(attn); attn=r; }
    { int p[3]={1,0,2}; mlx_array r=mlx_array_new(); mlx_transpose_axes(&r,attn,p,3,s); mlx_array_free(attn); attn=r; }
    { int sh[2]={T,H*D}; mlx_array r=mlx_array_new(); mlx_reshape(&r,attn,sh,2,s); mlx_array_free(attn); attn=r; }

    mlx_array attn_out = mlx_array_new();
    if (do_qmatmul_cached(&attn_out, attn, lw->o_w, lw->o_sc, lw->o_bi, m) != 0) { mlx_array_free(attn); return -1; }
    mlx_array_free(attn);

    mlx_array x1 = mlx_array_new();
    mlx_add(&x1, x, attn_out, s);
    mlx_array_free(attn_out);

    mlx_array h2 = mlx_array_new();
    if (do_rms_norm_cached(&h2, x1, lw->post_attn_norm_w, m) != 0) { mlx_array_free(x1); return -1; }

    mlx_array gate=mlx_array_new(), up=mlx_array_new();
    if (do_qmatmul_cached(&gate,h2,lw->gate_w,lw->gate_sc,lw->gate_bi,m)!=0 ||
        do_qmatmul_cached(&up,h2,lw->up_w,lw->up_sc,lw->up_bi,m)!=0) {
        mlx_array_free(h2); mlx_array_free(gate); mlx_array_free(up);
        mlx_array_free(x1); return -1; }
    mlx_array_free(h2);

    mlx_array sg=mlx_array_new();
    if (do_silu(&sg,gate,s)!=0) { mlx_array_free(gate); mlx_array_free(up); mlx_array_free(x1); return -1; }
    mlx_array_free(gate);

    mlx_array ffn=mlx_array_new();
    mlx_multiply(&ffn,sg,up,s);
    mlx_array_free(sg); mlx_array_free(up);

    mlx_array down=mlx_array_new();
    if (do_qmatmul_cached(&down,ffn,lw->down_w,lw->down_sc,lw->down_bi,m)!=0) { mlx_array_free(ffn); mlx_array_free(x1); return -1; }
    mlx_array_free(ffn);

    mlx_add(out_x, x1, down, s);
    mlx_array_free(x1); mlx_array_free(down);
    return 0;
}

/* ------------------------------------------------------------------
 * transformer_layer_decode
 *   Single-token forward pass using the pre-allocated KV cache buffer.
 *
 *   k_cache / v_cache are [KH, max_tokens, D] buffers.
 *   We write new K/V at position n_cached using mlx_slice_update
 *   (matching Python mlx-lm's cache.update_and_fetch pattern), then
 *   slice [:, :n_cached+1, :] for SDPA.
 *
 *   x must be [1, hidden].
 * ------------------------------------------------------------------ */

static int transformer_layer_decode(
    mlx_array* out_x,
    mlx_array  x,
    mlx_array* k_cache,   /* [KH, max_tokens, D] */
    mlx_array* v_cache,   /* [KH, max_tokens, D] */
    int layer, int n_cached, const llm_model_t* m)
{
    mlx_stream s = m->stream;
    int H  = m->num_heads;
    int KH = m->num_kv_heads;
    int D  = m->head_dim;
    const llm_layer_weights_t* lw = &m->layer_w[layer];

    mlx_array h = mlx_array_new();
    if (do_rms_norm_cached(&h, x, lw->input_norm_w, m) != 0) return -1;

    mlx_array Q=mlx_array_new(), K=mlx_array_new(), V=mlx_array_new();
    if (do_qmatmul_cached(&Q,h,lw->q_w,lw->q_sc,lw->q_bi,m)!=0 ||
        do_qmatmul_cached(&K,h,lw->k_w,lw->k_sc,lw->k_bi,m)!=0 ||
        do_qmatmul_cached(&V,h,lw->v_w,lw->v_sc,lw->v_bi,m)!=0) {
        mlx_array_free(h); mlx_array_free(Q); mlx_array_free(K); mlx_array_free(V); return -1; }
    mlx_array_free(h);

    /* Reshape [1, H*D] → [1, H, D]  (T=1 for new token) */
    { int sh[3]={1,H, D}; mlx_array r=mlx_array_new(); mlx_reshape(&r,Q,sh,3,s); mlx_array_free(Q); Q=r; }
    { int sh[3]={1,KH,D}; mlx_array r=mlx_array_new(); mlx_reshape(&r,K,sh,3,s); mlx_array_free(K); K=r; }
    { int sh[3]={1,KH,D}; mlx_array r=mlx_array_new(); mlx_reshape(&r,V,sh,3,s); mlx_array_free(V); V=r; }

    if (!is_null_arr(lw->q_norm_w)) {
        mlx_array r=mlx_array_new(); mlx_fast_rms_norm(&r,Q,lw->q_norm_w,m->rms_norm_eps,s); mlx_array_free(Q); Q=r; }
    if (!is_null_arr(lw->k_norm_w)) {
        mlx_array r=mlx_array_new(); mlx_fast_rms_norm(&r,K,lw->k_norm_w,m->rms_norm_eps,s); mlx_array_free(K); K=r; }

    /* Transpose [1, H/KH, D] → [H/KH, 1, D] */
    { int p[3]={1,0,2}; mlx_array r=mlx_array_new(); mlx_transpose_axes(&r,Q,p,3,s); mlx_array_free(Q); Q=r; }
    { int p[3]={1,0,2}; mlx_array r=mlx_array_new(); mlx_transpose_axes(&r,K,p,3,s); mlx_array_free(K); K=r; }
    { int p[3]={1,0,2}; mlx_array r=mlx_array_new(); mlx_transpose_axes(&r,V,p,3,s); mlx_array_free(V); V=r; }

    /* RoPE at position n_cached (the new token's position).
     * IMPORTANT: mlx rope requires 4D input [B, H, T, D] — with 3D [H, T, D]
     * it only rotates the first "head" and zeros the rest. */
    mlx_optional_float rope_base = { .value = m->rope_theta, .has_value = true };
    { mlx_array q4=mlx_array_new(); mlx_expand_dims(&q4, Q, 0, s); /* [1,H,1,D] */
      mlx_array nf=mlx_array_new(); mlx_array r=mlx_array_new();
      mlx_fast_rope(&r, q4, m->rope_dims, false, rope_base, 1.0f, n_cached, nf, s);
      mlx_array_free(nf); mlx_array_free(q4); mlx_array_free(Q);
      mlx_array q3=mlx_array_new(); mlx_squeeze_axis(&q3, r, 0, s); /* back to [H,1,D] */
      mlx_array_free(r); Q=q3; }
    { mlx_array k4=mlx_array_new(); mlx_expand_dims(&k4, K, 0, s); /* [1,KH,1,D] */
      mlx_array nf=mlx_array_new(); mlx_array r=mlx_array_new();
      mlx_fast_rope(&r, k4, m->rope_dims, false, rope_base, 1.0f, n_cached, nf, s);
      mlx_array_free(nf); mlx_array_free(k4); mlx_array_free(K);
      mlx_array k3=mlx_array_new(); mlx_squeeze_axis(&k3, r, 0, s); /* back to [KH,1,D] */
      mlx_array_free(r); K=k3; }

    /* Write new K/V into the cache buffer at position n_cached. */
    {
        int st[3] = {0, n_cached, 0};
        int sp[3] = {KH, n_cached+1, D};
        int sr[3] = {1, 1, 1};
        mlx_array new_k = mlx_array_new();
        mlx_slice_update(&new_k, *k_cache, K, st, 3, sp, 3, sr, 3, s);
        mlx_array_free(*k_cache); *k_cache = new_k;
        mlx_array_free(K);
    }
    {
        int st[3] = {0, n_cached, 0};
        int sp[3] = {KH, n_cached+1, D};
        int sr[3] = {1, 1, 1};
        mlx_array new_v = mlx_array_new();
        mlx_slice_update(&new_v, *v_cache, V, st, 3, sp, 3, sr, 3, s);
        mlx_array_free(*v_cache); *v_cache = new_v;
        mlx_array_free(V);
    }

    /* Extract K/V up to current position for SDPA:
     *   K_active = k_cache[:, :n_cached+1, :]  [KH, T, D]
     * This matches Python's: return self.keys[..., :self.offset, :] */
    mlx_array K_active=mlx_array_new(), V_active=mlx_array_new();
    {
        int st[3] = {0, 0, 0};
        int sp[3] = {KH, n_cached+1, D};
        int sr[3] = {1, 1, 1};
        mlx_slice(&K_active, *k_cache, st, 3, sp, 3, sr, 3, s);
        mlx_slice(&V_active, *v_cache, st, 3, sp, 3, sr, 3, s);
    }

    /* Expand batch dim for SDPA: Q→[1,H,1,D], K→[1,KH,T,D] */
    mlx_array Qb=mlx_array_new(), Kb=mlx_array_new(), Vb=mlx_array_new();
    mlx_expand_dims(&Qb, Q,        0, s); mlx_array_free(Q);
    mlx_expand_dims(&Kb, K_active, 0, s); mlx_array_free(K_active);
    mlx_expand_dims(&Vb, V_active, 0, s); mlx_array_free(V_active);

    /* SDPA — no causal mask for T_q=1 */
    float scale = 1.0f / sqrtf((float)D);
    mlx_array attn=mlx_array_new();
    { mlx_array nm=mlx_array_new(); mlx_array ns=mlx_array_new();
      int err=mlx_fast_scaled_dot_product_attention(&attn,Qb,Kb,Vb,scale,"",nm,ns,s);
      mlx_array_free(nm); mlx_array_free(ns);
      mlx_array_free(Qb); mlx_array_free(Kb); mlx_array_free(Vb);
      if (err!=0) return -1; }

    /* Remove batch → [H,1,D]; transpose → [1,H,D]; reshape → [1, H*D] */
    { mlx_array r=mlx_array_new(); mlx_squeeze_axis(&r,attn,0,s); mlx_array_free(attn); attn=r; }
    { int p[3]={1,0,2}; mlx_array r=mlx_array_new(); mlx_transpose_axes(&r,attn,p,3,s); mlx_array_free(attn); attn=r; }
    { int sh[2]={1,H*D}; mlx_array r=mlx_array_new(); mlx_reshape(&r,attn,sh,2,s); mlx_array_free(attn); attn=r; }

    mlx_array attn_out=mlx_array_new();
    if (do_qmatmul_cached(&attn_out,attn,lw->o_w,lw->o_sc,lw->o_bi,m)!=0) { mlx_array_free(attn); return -1; }
    mlx_array_free(attn);

    mlx_array x1=mlx_array_new();
    mlx_add(&x1, x, attn_out, s);
    mlx_array_free(attn_out);

    mlx_array h2=mlx_array_new();
    if (do_rms_norm_cached(&h2,x1,lw->post_attn_norm_w,m)!=0) { mlx_array_free(x1); return -1; }

    mlx_array gate=mlx_array_new(), up=mlx_array_new();
    if (do_qmatmul_cached(&gate,h2,lw->gate_w,lw->gate_sc,lw->gate_bi,m)!=0 ||
        do_qmatmul_cached(&up,h2,lw->up_w,lw->up_sc,lw->up_bi,m)!=0) {
        mlx_array_free(h2); mlx_array_free(gate); mlx_array_free(up); mlx_array_free(x1); return -1; }
    mlx_array_free(h2);

    /* SwiGLU via compiled closure: [gate, up] → [silu(gate) * up]
     * Fused kernel eliminates 2 of 3 GPU dispatches per layer. */
    mlx_vector_array swiglu_in = mlx_vector_array_new();
    mlx_vector_array_append_value(swiglu_in, gate);
    mlx_vector_array_append_value(swiglu_in, up);
    mlx_array_free(gate); mlx_array_free(up);

    mlx_vector_array swiglu_out = mlx_vector_array_new();
    if (mlx_closure_apply(&swiglu_out, m->swiglu_compiled, swiglu_in) != 0) {
        mlx_vector_array_free(swiglu_in); mlx_vector_array_free(swiglu_out);
        mlx_array_free(x1); return -1;
    }
    mlx_vector_array_free(swiglu_in);

    mlx_array ffn=mlx_array_new();
    mlx_vector_array_get(&ffn, swiglu_out, 0);
    mlx_vector_array_free(swiglu_out);

    mlx_array down=mlx_array_new();
    if (do_qmatmul_cached(&down,ffn,lw->down_w,lw->down_sc,lw->down_bi,m)!=0) { mlx_array_free(ffn); mlx_array_free(x1); return -1; }
    mlx_array_free(ffn);

    mlx_add(out_x, x1, down, s);
    mlx_array_free(x1); mlx_array_free(down);
    return 0;
}

/* Build LM-head graph: x [1, hidden] → unevaluated logits32 (float32).
 * Uses cached m->lm_head_w/sc/bi pointers. */
static int do_lm_head_graph(mlx_array* out_logits32, mlx_array x,
                             const llm_model_t* m)
{
    mlx_stream s = m->stream;
    mlx_array logits = mlx_array_new();
    if (m->lm_head_mode == 2 || m->lm_head_mode == 3) {
        if (do_qmatmul_cached(&logits, x, m->lm_head_w, m->lm_head_sc, m->lm_head_bi, m) != 0) return -1;
    } else {
        /* mode 0 or 1: float matmul with embed or lm_head weight */
        if (do_fmatmul_t(&logits, x, m->lm_head_w, s) != 0) return -1;
    }
    mlx_astype(out_logits32, logits, MLX_FLOAT32, s);
    mlx_array_free(logits);
    return 0;
}

/* Copy evaluated logits32 to CPU float array */
static int do_copy_logits(float* out_logits, int vocab_size,
                           mlx_array logits32)
{
    const float* data = mlx_array_data_float32(logits32);
    if (!data) return -1;
    memcpy(out_logits, data, (size_t)vocab_size * sizeof(float));
    return 0;
}

/* Shared LM-head helper: x [1, hidden] → logits float array (eval + copy) */
static int do_lm_head_and_copy(float* out_logits, int vocab_size,
                                mlx_array x, const llm_model_t* m)
{
    mlx_stream s = m->stream;
    mlx_array logits32 = mlx_array_new();
    if (do_lm_head_graph(&logits32, x, m) != 0) return -1;
    mlx_array_eval(logits32);
    mlx_synchronize(s);
    int err = do_copy_logits(out_logits, vocab_size, logits32);
    mlx_array_free(logits32);
    return err;
}

/* ------------------------------------------------------------------
 * llm_prefill
 * ------------------------------------------------------------------ */

llm_cache_t* llm_prefill(
    llm_model_t* m,
    const int32_t* token_ids,
    int n_tokens,
    float* out_logits,
    int vocab_size)
{
    mlx_stream s = m->stream;
    int KH = m->num_kv_heads;
    int D  = m->head_dim;

    /* Allocate cache struct — k/v arrays will be set up below */
    llm_cache_t* cache = (llm_cache_t*)calloc(1, sizeof(llm_cache_t));
    if (!cache) return NULL;
    cache->n_layers = m->num_layers;
    cache->n_tokens = n_tokens;
    cache->pending_logits = mlx_array_new();
    cache->eval_batch = mlx_vector_array_new();
    cache->k = (mlx_array*)calloc(m->num_layers, sizeof(mlx_array));
    cache->v = (mlx_array*)calloc(m->num_layers, sizeof(mlx_array));
    if (!cache->k || !cache->v) { llm_cache_free(cache); return NULL; }

    /* Token embedding */
    mlx_array x = mlx_array_new();
    if (do_embed_lookup(&x, token_ids, n_tokens, m) != 0) {
        llm_cache_free(cache); return NULL; }

    /* Run transformer layers, collecting raw K/V [KH, T, D] per layer */
    for (int i = 0; i < m->num_layers; i++) {
        mlx_array x_new = mlx_array_new();
        int err = transformer_layer_prefill(
            &x_new, &cache->k[i], &cache->v[i], x, i, n_tokens, m);
        mlx_array_free(x);
        if (err != 0) { llm_cache_free(cache); return NULL; }
        x = x_new;
    }

    /* Convert K/V from lazy [KH, T, D] to pre-allocated [KH, max_tokens, D]
     * buffers. Skip intermediate materialization — let slice_update work on
     * lazy arrays and eval everything once at the end. */
    int max_tokens = n_tokens + 1024;
    cache->max_tokens = max_tokens;
    /* dtype is a graph property — works on lazy arrays, no eval needed */
    mlx_dtype kv_dtype = mlx_array_dtype(cache->k[0]);

    for (int i = 0; i < m->num_layers; i++) {
        mlx_array K_prompt = cache->k[i];  /* [KH, T, D], lazy */
        mlx_array V_prompt = cache->v[i];

        int buf_sh[3] = {KH, max_tokens, D};
        int st[3] = {0, 0, 0};
        int sp[3] = {KH, n_tokens, D};
        int sr[3] = {1, 1, 1};

        mlx_array k_buf = mlx_array_new();
        mlx_zeros(&k_buf, buf_sh, 3, kv_dtype, s);
        mlx_array k_new = mlx_array_new();
        mlx_slice_update(&k_new, k_buf, K_prompt, st, 3, sp, 3, sr, 3, s);
        mlx_array_free(k_buf); mlx_array_free(K_prompt);
        cache->k[i] = k_new;

        mlx_array v_buf = mlx_array_new();
        mlx_zeros(&v_buf, buf_sh, 3, kv_dtype, s);
        mlx_array v_new = mlx_array_new();
        mlx_slice_update(&v_new, v_buf, V_prompt, st, 3, sp, 3, sr, 3, s);
        mlx_array_free(v_buf); mlx_array_free(V_prompt);
        cache->v[i] = v_new;
    }

    /* Batch eval: single mlx_eval for all K/V buffers */
    {
        mlx_vector_array to_eval = mlx_vector_array_new();
        for (int i = 0; i < m->num_layers; i++) {
            mlx_vector_array_append_value(to_eval, cache->k[i]);
            mlx_vector_array_append_value(to_eval, cache->v[i]);
        }
        mlx_eval(to_eval);
        mlx_vector_array_free(to_eval);
    }

    /* Final norm → slice last token → LM head */
    { mlx_array xn=mlx_array_new();
      if (do_rms_norm_cached(&xn, x, m->final_norm_w, m) != 0) {
          mlx_array_free(x); llm_cache_free(cache); return NULL; }
      mlx_array_free(x); x = xn; }

    mlx_array last = mlx_array_new();
    { int st[2]={n_tokens-1,0}, sp[2]={n_tokens,m->hidden_size}, sr[2]={1,1};
      mlx_slice(&last, x, st, 2, sp, 2, sr, 2, s);
      mlx_array_free(x); }

    if (do_lm_head_and_copy(out_logits, vocab_size, last, m) != 0) {
        mlx_array_free(last); llm_cache_free(cache); return NULL; }
    mlx_array_free(last);

    return cache;
}

/* ------------------------------------------------------------------
 * llm_decode_step — Phase 1: build graph + async eval (returns immediately)
 * ------------------------------------------------------------------ */

int llm_decode_step(
    llm_model_t* m,
    llm_cache_t* cache,
    int32_t token_id)
{
    int n_cached = cache->n_tokens;

    /* Embed the single new token → [1, hidden] */
    mlx_array x = mlx_array_new();
    if (do_embed_lookup(&x, &token_id, 1, m) != 0) return -1;

    /* Run all decode layers — each writes one position into the K/V buffers */
    for (int i = 0; i < m->num_layers; i++) {
        mlx_array x_new = mlx_array_new();
        int err = transformer_layer_decode(
            &x_new, x, &cache->k[i], &cache->v[i], i, n_cached, m);
        mlx_array_free(x);
        if (err != 0) return -1;
        x = x_new;
    }

    cache->n_tokens = n_cached + 1;

    /* Final norm (x is already [1, hidden]) → LM head (graph only) */
    { mlx_array xn=mlx_array_new();
      if (do_rms_norm_cached(&xn, x, m->final_norm_w, m) != 0) { mlx_array_free(x); return -1; }
      mlx_array_free(x); x = xn; }

    mlx_array logits32 = mlx_array_new();
    if (do_lm_head_graph(&logits32, x, m) != 0) { mlx_array_free(x); return -1; }
    mlx_array_free(x);

    /* Store pending logits for llm_decode_read */
    mlx_array_free(cache->pending_logits);
    cache->pending_logits = logits32;

    /* Async eval: K/V caches + logits — GPU starts, returns immediately.
     * decode_read() will call mlx_array_eval to wait for results.
     * Reuse eval_batch vector to avoid per-step allocation. */
    mlx_vector_array_free(cache->eval_batch);
    cache->eval_batch = mlx_vector_array_new();
    for (int i = 0; i < m->num_layers; i++) {
        mlx_vector_array_append_value(cache->eval_batch, cache->k[i]);
        mlx_vector_array_append_value(cache->eval_batch, cache->v[i]);
    }
    mlx_vector_array_append_value(cache->eval_batch, logits32);
    mlx_async_eval(cache->eval_batch);

    return 0;
}

/* ------------------------------------------------------------------
 * llm_decode_read — Phase 2: wait for GPU results + copy logits to CPU
 * ------------------------------------------------------------------ */

int llm_decode_read(
    llm_model_t* m,
    llm_cache_t* cache,
    float* out_logits,
    int vocab_size)
{
    (void)m;
    mlx_array logits32 = cache->pending_logits;
    /* Wait for async eval to complete. mlx_array_eval detects the array
     * is already scheduled and calls .wait() — no double evaluation. */
    mlx_array_eval(logits32);
    int err = do_copy_logits(out_logits, vocab_size, logits32);
    /* Reset pending */
    mlx_array_free(cache->pending_logits);
    cache->pending_logits = mlx_array_new();
    return err;
}

/* ------------------------------------------------------------------
 * llm_decode — backward-compatible synchronous decode (step + read)
 * ------------------------------------------------------------------ */

int llm_decode(
    llm_model_t* m,
    llm_cache_t* cache,
    int32_t token_id,
    float* out_logits,
    int vocab_size)
{
    if (llm_decode_step(m, cache, token_id) != 0) return -1;
    return llm_decode_read(m, cache, out_logits, vocab_size);
}

/* ------------------------------------------------------------------
 * decode_and_sample — build full decode graph + GPU-side sampling.
 *
 * token_arr: scalar int32 mlx_array (can be unevaluated/lazy)
 * out_sampled: lazy scalar int32 — the sampled next token
 *
 * Builds the entire decode graph from embedding through LM head and
 * sampling, keeping everything on GPU. The key insight: when token_arr
 * is a lazy result from a previous decode_and_sample, MLX builds a
 * continuation graph without evaluating anything.
 * ------------------------------------------------------------------ */

static int decode_and_sample(
    llm_model_t* m, llm_cache_t* cache,
    mlx_array token_arr,
    float temperature,
    mlx_array* out_sampled)
{
    mlx_stream s = m->stream;
    int n_cached = cache->n_tokens;

    /* 1. Lazy embedding lookup: [1, hidden] */
    mlx_array x = mlx_array_new();
    if (do_embed_lookup_array(&x, token_arr, m) != 0) return -1;

    /* 2. All transformer layers (each writes one position into KV cache) */
    for (int i = 0; i < m->num_layers; i++) {
        mlx_array x_new = mlx_array_new();
        int err = transformer_layer_decode(
            &x_new, x, &cache->k[i], &cache->v[i], i, n_cached, m);
        mlx_array_free(x);
        if (err != 0) return -1;
        x = x_new;
    }
    cache->n_tokens = n_cached + 1;

    /* 3. Final norm → LM head → logits32 [1, vocab_size] */
    { mlx_array xn = mlx_array_new();
      if (do_rms_norm_cached(&xn, x, m->final_norm_w, m) != 0) {
          mlx_array_free(x); return -1; }
      mlx_array_free(x); x = xn; }

    mlx_array logits32 = mlx_array_new();
    if (do_lm_head_graph(&logits32, x, m) != 0) {
        mlx_array_free(x); return -1; }
    mlx_array_free(x);

    /* 4. GPU-side sampling */
    mlx_array sampled = mlx_array_new();
    if (temperature <= 0.0f) {
        /* Greedy: argmax over last axis */
        if (mlx_argmax_axis(&sampled, logits32, -1, false, s) != 0) {
            mlx_array_free(logits32); return -1; }
    } else {
        /* Stochastic: scale logits then categorical sample */
        mlx_array temp_scalar = mlx_array_new_float(temperature);
        mlx_array scaled = mlx_array_new();
        mlx_divide(&scaled, logits32, temp_scalar, s);
        mlx_array_free(temp_scalar);
        /* mlx_random_categorical expects unnormalized log-probs (logits) */
        mlx_array null_key = mlx_array_new();
        if (mlx_random_categorical(&sampled, scaled, -1, null_key, s) != 0) {
            mlx_array_free(scaled); mlx_array_free(null_key);
            mlx_array_free(logits32); return -1; }
        mlx_array_free(scaled); mlx_array_free(null_key);
        /* categorical returns float — cast to int32 */
        mlx_array sampled_i32 = mlx_array_new();
        mlx_astype(&sampled_i32, sampled, MLX_INT32, s);
        mlx_array_free(sampled);
        sampled = sampled_i32;
    }
    mlx_array_free(logits32);

    /* 5. Async eval: K/V caches + sampled token */
    {
        mlx_vector_array_free(cache->eval_batch);
        cache->eval_batch = mlx_vector_array_new();
        for (int i = 0; i < m->num_layers; i++) {
            mlx_vector_array_append_value(cache->eval_batch, cache->k[i]);
            mlx_vector_array_append_value(cache->eval_batch, cache->v[i]);
        }
        mlx_vector_array_append_value(cache->eval_batch, sampled);
        mlx_async_eval(cache->eval_batch);
    }

    *out_sampled = sampled;
    return 0;
}

/* ------------------------------------------------------------------
 * llm_pipeline_generate — double-buffered generation with GPU sampling.
 *
 * Mirrors Python mlx-lm's exact decode pattern:
 *   1. Build step N+1's graph using step N's lazy (unevaluated) token
 *   2. GPU evaluates step N while CPU constructs step N+1's graph
 *   3. Block on step N's result only after step N+1 is scheduled
 *
 * on_token callback: return non-zero to stop generation early.
 * Returns 0 on success, non-zero on error.
 * ------------------------------------------------------------------ */

/* C++ native decode loop — declared in mlx_llm_cpp.cpp */
extern int llm_pipeline_generate_cpp(
    void* model, void* cache,
    int32_t first_token_id, int max_tokens, int eos_id, float temperature,
    int (*on_token)(int32_t token_id, void* ctx), void* ctx);

int llm_pipeline_generate(
    llm_model_t* model,
    llm_cache_t* cache,
    int32_t      first_token_id,
    int          max_tokens,
    int          eos_id,
    float        temperature,
    int        (*on_token)(int32_t token_id, void* ctx),
    void*        ctx)
{
    /* Delegate to C++ native decode loop with compiled SwiGLU.
     * Uses MLX C++ API directly — same performance as C wrapper path
     * but cleaner code with native array semantics. */
    return llm_pipeline_generate_cpp(
        (void*)model, (void*)cache,
        first_token_id, max_tokens, eos_id, temperature,
        on_token, ctx);
}

/* ------------------------------------------------------------------
 * llm_cache_free
 * ------------------------------------------------------------------ */

void llm_cache_free(llm_cache_t* cache) {
    if (!cache) return;
    mlx_array_free(cache->pending_logits);
    mlx_vector_array_free(cache->eval_batch);
    if (cache->k) {
        for (int i = 0; i < cache->n_layers; i++) mlx_array_free(cache->k[i]);
        free(cache->k);
    }
    if (cache->v) {
        for (int i = 0; i < cache->n_layers; i++) mlx_array_free(cache->v[i]);
        free(cache->v);
    }
    free(cache);
}

/* ------------------------------------------------------------------
 * llm_free
 * ------------------------------------------------------------------ */

void llm_free(llm_model_t* m) {
    if (!m) return;
    /* Free cached weight pointers */
    if (m->layer_w) {
        for (int i = 0; i < m->num_layers; i++) {
            llm_layer_weights_t* lw = &m->layer_w[i];
            mlx_array_free(lw->q_w);  mlx_array_free(lw->q_sc);  mlx_array_free(lw->q_bi);
            mlx_array_free(lw->k_w);  mlx_array_free(lw->k_sc);  mlx_array_free(lw->k_bi);
            mlx_array_free(lw->v_w);  mlx_array_free(lw->v_sc);  mlx_array_free(lw->v_bi);
            mlx_array_free(lw->o_w);  mlx_array_free(lw->o_sc);  mlx_array_free(lw->o_bi);
            mlx_array_free(lw->input_norm_w);
            mlx_array_free(lw->q_norm_w);
            mlx_array_free(lw->k_norm_w);
            mlx_array_free(lw->post_attn_norm_w);
            mlx_array_free(lw->gate_w); mlx_array_free(lw->gate_sc); mlx_array_free(lw->gate_bi);
            mlx_array_free(lw->up_w);   mlx_array_free(lw->up_sc);   mlx_array_free(lw->up_bi);
            mlx_array_free(lw->down_w); mlx_array_free(lw->down_sc); mlx_array_free(lw->down_bi);
        }
        free(m->layer_w);
    }
    mlx_array_free(m->embed_w);  mlx_array_free(m->embed_sc);  mlx_array_free(m->embed_bi);
    mlx_array_free(m->final_norm_w);
    mlx_array_free(m->lm_head_w); mlx_array_free(m->lm_head_sc); mlx_array_free(m->lm_head_bi);
    mlx_closure_free(m->swiglu_compiled);
    mlx_map_string_to_array_free(m->weights);
    mlx_map_string_to_string_free(m->metadata);
    mlx_stream_free(m->stream);
    free(m);
}
