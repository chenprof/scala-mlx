import scala.scalanative.unsafe.*
import scala.scalanative.libc.stdlib

/** Wraps the C llm_model_t and drives the autoregressive generation loop.
  *
  * Two decode paths:
  * - Pipeline (topP >= 1.0): GPU-side sampling + double-buffered decode
  *   Eliminates 512KB logits memcpy/token, GPU never idles during graph build.
  * - CPU sampling (topP < 1.0): existing decode_step/read with nucleus filter.
  */
class LlamaModel(cfg: LlamaConfig, modelPath: os.Path) extends AutoCloseable:

  // Pre-allocated sampler — holds Float logits buffer + Long sort scratch.
  // Reused every token step to eliminate per-step GC pressure.
  private val sampler = new Sampler(cfg.vocabSize)

  private val handle: MlxLlm.ModelHandle =
    val z = Zone.open()
    try
      given Zone = z
      MlxLlm.llm_load(
        toCString(modelPath.toString),
        cfg.numHiddenLayers,
        cfg.hiddenSize,
        cfg.numAttentionHeads,
        cfg.numKeyValueHeads,
        cfg.headDim,
        cfg.intermediateSize,
        cfg.vocabSize,
        cfg.rmsNormEps,
        cfg.ropeTheta,
        cfg.ropeDims
      )
    finally z.close()

  if handle == null then
    throw RuntimeException(s"Failed to load model from $modelPath")

  /* ----------------------------------------------------------------
   * Generate — KV-cache path (O(n) per decode step)
   *
   * 1. llm_prefill processes the full prompt, populates the KV cache,
   *    and returns logits for the first generated token.
   * 2. Decode via pipeline (GPU sampling) or CPU sampling path.
   *
   * temperature: 0.0 = greedy argmax; >0 = stochastic sampling
   * topP:        1.0 = no nucleus filter; <1.0 = nucleus sampling
   * rng:         random number generator (seed for reproducibility)
   * onToken:     called for each newly generated token id (streaming)
   * Returns the list of generated (non-prompt) token ids.
   * ---------------------------------------------------------------- */
  def generate(
      promptTokens: Array[Int],
      maxNewTokens: Int,
      eosId:        Int,
      temperature:  Float                = 0.0f,
      topP:         Float                = 1.0f,
      rng:          scala.util.Random    = scala.util.Random(),
      onToken:      Int => Unit          = _ => ()
  ): Array[Int] =

    val vocabSize = cfg.vocabSize

    // Single C buffer for logits — reused by prefill and every decode step.
    val logitsBuf = stdlib.malloc(vocabSize.toLong * 4L).asInstanceOf[Ptr[CFloat]]
    if logitsBuf == null then throw RuntimeException("Failed to allocate logits buffer")

    var cache: MlxLlm.CacheHandle = null
    try
      val generated = scala.collection.mutable.ArrayBuffer.empty[Int]

      // ── Prefill: full prompt → KV cache + first token logits ──────────────
      cache = {
        val z = Zone.open()
        try
          given Zone = z
          val buf = alloc[CInt](promptTokens.length)
          for i <- promptTokens.indices do buf(i) = promptTokens(i)
          MlxLlm.llm_prefill(handle, buf, promptTokens.length, logitsBuf, vocabSize)
        finally z.close()
      }
      if cache == null then throw RuntimeException("llm_prefill failed")

      val firstTok = sampler.sample(logitsBuf, temperature, topP, rng)
      onToken(firstTok)
      generated += firstTok

      if firstTok != eosId && generated.length < maxNewTokens then
        if topP >= 1.0f then
          // ── Pipeline: GPU sampling + double-buffered decode ─────────────
          // Tokens are collected in a C buffer via callback, then delivered
          // to onToken after pipeline completes. GPU-side argmax/categorical
          // eliminates 512KB logits memcpy per token. Double buffering
          // overlaps GPU eval(N) with CPU graph-build(N+1).
          val remaining = maxNewTokens - 1

          // Allocate C buffer for token collection
          val tokenBuf = stdlib.malloc(remaining.toLong * 4L).asInstanceOf[Ptr[CInt]]
          // ctx layout: [Ptr[CInt] tokenBuf (8 bytes), CInt count (4 bytes)]
          val ctx = stdlib.malloc(16L)
          !(ctx.asInstanceOf[Ptr[Ptr[CInt]]]) = tokenBuf
          !((ctx + 8).asInstanceOf[Ptr[CInt]]) = 0

          val rc = MlxLlm.llm_pipeline_generate(
            handle, cache, firstTok, remaining, eosId, temperature,
            LlamaModel.pipelineTokenCb, ctx
          )

          // Deliver collected tokens via onToken callback
          val count = !((ctx + 8).asInstanceOf[Ptr[CInt]])
          var i = 0
          while i < count do
            val tok = tokenBuf(i)
            onToken(tok)
            generated += tok
            i += 1

          stdlib.free(ctx)
          stdlib.free(tokenBuf.asInstanceOf[Ptr[Byte]])

          if rc != 0 then throw RuntimeException("llm_pipeline_generate failed")

        else
          // ── CPU sampling with nucleus filter ────────────────────────────
          // Kick off first decode step (GPU starts working, returns immediately)
          val rc0 = MlxLlm.llm_decode_step(handle, cache, firstTok)
          if rc0 != 0 then throw RuntimeException(s"llm_decode_step failed (rc=$rc0)")

          var continue_ = true
          while continue_ do
            // Wait for GPU result from previous step
            val rcR = MlxLlm.llm_decode_read(handle, cache, logitsBuf, vocabSize)
            if rcR != 0 then throw RuntimeException(s"llm_decode_read failed (rc=$rcR)")

            val tok = sampler.sample(logitsBuf, temperature, topP, rng)
            onToken(tok)
            generated += tok

            if tok != eosId && generated.length < maxNewTokens then
              // Start next decode step (GPU works while we loop back)
              val rcS = MlxLlm.llm_decode_step(handle, cache, tok)
              if rcS != 0 then throw RuntimeException(s"llm_decode_step failed (rc=$rcS)")
            else
              continue_ = false

      generated.toArray

    finally
      if cache != null then MlxLlm.llm_cache_free(cache)
      stdlib.free(logitsBuf.asInstanceOf[Ptr[Byte]])

  override def close(): Unit =
    if handle != null then MlxLlm.llm_free(handle)

object LlamaModel:
  /** C callback for llm_pipeline_generate: stores each token into a buffer.
    * ctx layout: [Ptr[CInt] tokenBuf (8 bytes), CInt count (4 bytes)] */
  private val pipelineTokenCb: CFuncPtr2[CInt, Ptr[Byte], CInt] =
    (tok: CInt, ctx: Ptr[Byte]) => {
      val buf = !(ctx.asInstanceOf[Ptr[Ptr[CInt]]])
      val countP = (ctx + 8).asInstanceOf[Ptr[CInt]]
      val i = !countP
      buf(i) = tok
      !countP = i + 1
      0
    }
