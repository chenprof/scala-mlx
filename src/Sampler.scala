/** Zero-allocation sampler: allocates all scratch space once at construction.
  *
  * Stateless [[Sampling]] functions allocate per-call; this class pre-allocates
  * a Long scratch buffer (for indirect sort) and a Float logits buffer, then
  * reuses them every token — eliminating ~3 MB of GC pressure per step.
  *
  * Call site (LlamaModel):
  *   val sampler = new Sampler(cfg.vocabSize)
  *   val nextTok = sampler.sample(logitsBuf, temperature, topP, rng)
  */
class Sampler(vocabSize: Int):

  /** Pre-allocated Float buffer — copy of the C logits for in-place ops. */
  val logits = new Array[Float](vocabSize)

  /** Pre-allocated Long scratch for indirect sort: packed (negBits | index). */
  private val packed = new Array[Long](vocabSize)

  /** Sample the next token.
    *
    * Copies [[logitsBuf]] into [[logits]] then works in-place — no allocations.
    * temperature=0.0 → greedy argmax.
    * topP=1.0 → full-distribution multinomial.
    */
  def sample(
      logitsBuf:   scala.scalanative.unsafe.Ptr[scala.scalanative.unsafe.CFloat],
      temperature: Float,
      topP:        Float,
      rng:         scala.util.Random
  ): Int =
    // Copy C buffer → Scala array
    var i = 0
    while i < vocabSize do { logits(i) = logitsBuf(i); i += 1 }

    if temperature == 0.0f then return Sampling.argmax(logits)

    // Temperature scale in-place (multiply avoids one float division chain)
    val invTemp = 1.0f / temperature
    i = 0
    while i < vocabSize do { logits(i) *= invTemp; i += 1 }

    // Numerically-stable softmax in-place
    var maxVal = Float.NegativeInfinity
    i = 0
    while i < vocabSize do { if logits(i) > maxVal then maxVal = logits(i); i += 1 }
    var softmaxSum = 0.0f
    i = 0
    while i < vocabSize do
      val e = math.exp((logits(i) - maxVal).toDouble).toFloat
      logits(i) = e
      softmaxSum += e
      i += 1
    i = 0
    while i < vocabSize do { logits(i) /= softmaxSum; i += 1 }

    if topP >= 1.0f then
      return Sampling.sampleFromProbs(logits, rng)

    // Indirect sort via Long packing — zero object allocations.
    // Pack (-prob bits | index) so java.util.Arrays.sort(long[]) sorts by
    // descending probability without boxing a single Int.
    i = 0
    while i < vocabSize do
      val negBits = java.lang.Float.floatToRawIntBits(-logits(i)).toLong
      packed(i) = (negBits << 32) | (i.toLong & 0xFFFFFFFFL)
      i += 1
    java.util.Arrays.sort(packed)

    // Walk sorted nucleus until cumulative prob ≥ topP
    var cumProb = 0.0f
    var cutoff  = vocabSize
    i = 0
    while i < vocabSize do
      val idx = (packed(i) & 0xFFFFFFFFL).toInt
      cumProb += logits(idx)
      if cumProb >= topP && cutoff == vocabSize then cutoff = i + 1
      i += 1

    // Sample directly from sorted nucleus — no filtered-array construction
    val r = rng.nextFloat() * cumProb
    var sampledCum = 0.0f
    i = 0
    while i < cutoff - 1 do
      val idx = (packed(i) & 0xFFFFFFFFL).toInt
      sampledCum += logits(idx)
      if r < sampledCum then return idx
      i += 1
    (packed(cutoff - 1) & 0xFFFFFFFFL).toInt  // fallback: last nucleus token
