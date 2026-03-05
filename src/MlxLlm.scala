import scala.scalanative.unsafe.*

/** Thin FFI bindings for the three functions exposed by native/mlx_llm.c.
  * All heavy lifting (tensor ops, GPU dispatch) happens on the C side.
  */
@extern
object MlxLlm:

  /** Opaque handle to an llm_model_t allocated in C. */
  type ModelHandle = Ptr[Byte]

  /** Load a quantized LLaMA model from a SafeTensors file.
    * Returns null on failure. */
  def llm_load(
      path: CString,
      numLayers: CInt,
      hiddenSize: CInt,
      numHeads: CInt,
      numKvHeads: CInt,
      headDim: CInt,
      intermediateSize: CInt,
      vocabSize: CInt,
      rmsNormEps: CFloat,
      ropeTheta: CFloat,
      ropeDims: CInt
  ): ModelHandle = extern

  /** Forward pass over the full token sequence token_ids[0..nTokens-1].
    * Writes vocab_size floats (logits for the next token) into outLogits.
    * Returns 0 on success. */
  def llm_forward(
      model: ModelHandle,
      tokenIds: Ptr[CInt],
      nTokens: CInt,
      outLogits: Ptr[CFloat],
      vocabSize: CInt
  ): CInt = extern

  /** Free a model and all its GPU resources. */
  def llm_free(model: ModelHandle): Unit = extern

  // ── KV-cache API ─────────────────────────────────────────────────────────

  /** Opaque handle to an llm_cache_t (per-layer K/V arrays on GPU). */
  type CacheHandle = Ptr[Byte]

  /** Prefill: process full prompt, populate KV cache, write next-token logits.
    * Returns null on failure. */
  def llm_prefill(
      model:     ModelHandle,
      tokenIds:  Ptr[CInt],
      nTokens:   CInt,
      outLogits: Ptr[CFloat],
      vocabSize: CInt
  ): CacheHandle = extern

  /** Decode: process one new token using the KV cache, write next-token logits.
    * Updates the cache in-place. Returns 0 on success. */
  def llm_decode(
      model:     ModelHandle,
      cache:     CacheHandle,
      tokenId:   CInt,
      outLogits: Ptr[CFloat],
      vocabSize: CInt
  ): CInt = extern

  /** Decode step (async): build graph + start GPU eval, return immediately.
    * Call llm_decode_read() to retrieve the results. Returns 0 on success. */
  def llm_decode_step(
      model:   ModelHandle,
      cache:   CacheHandle,
      tokenId: CInt
  ): CInt = extern

  /** Decode read: wait for GPU results from previous decode_step, copy logits.
    * Blocks until GPU computation is complete. Returns 0 on success. */
  def llm_decode_read(
      model:     ModelHandle,
      cache:     CacheHandle,
      outLogits: Ptr[CFloat],
      vocabSize: CInt
  ): CInt = extern

  /** Double-buffered generation with GPU-side sampling.
    * Prefill must be called first. on_token callback returns non-zero to stop.
    * Returns 0 on success. */
  def llm_pipeline_generate(
      model:       ModelHandle,
      cache:       CacheHandle,
      firstTokenId: CInt,
      maxTokens:   CInt,
      eosId:       CInt,
      temperature: CFloat,
      onToken:     CFuncPtr2[CInt, Ptr[Byte], CInt],
      ctx:         Ptr[Byte]
  ): CInt = extern

  /** Free a KV cache and its GPU tensors. */
  def llm_cache_free(cache: CacheHandle): Unit = extern
