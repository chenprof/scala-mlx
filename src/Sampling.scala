/** Sampling strategies for autoregressive text generation.
  *
  * Provides greedy and stochastic sampling over a logit distribution.
  * Extracted from LlamaModel so it can be unit-tested without FFI.
  *
  * Typical call site (in LlamaModel):
  *   val nextTok = Sampling.sample(logits, temperature, topP, rng)
  */
object Sampling:

  /** Greedy argmax: return the index of the highest logit. */
  def argmax(logits: Array[Float]): Int =
    var bestId    = 0
    var bestScore = Float.NegativeInfinity
    var i         = 0
    while i < logits.length do
      if logits(i) > bestScore then { bestScore = logits(i); bestId = i }
      i += 1
    bestId

  /** Divide all logits by temperature in-place, returning the same array.
    * Higher temperature → flatter distribution; lower → sharper.
    * Caller must not pass temperature=0 (use [[sample]] which short-circuits
    * to [[argmax]] for temperature=0). */
  def applyTemperature(logits: Array[Float], temperature: Float): Array[Float] =
    var i = 0
    while i < logits.length do
      logits(i) /= temperature
      i += 1
    logits

  /** Numerically stable softmax: converts logits to a probability distribution.
    * Subtracts max before exp to prevent overflow. Returns a new array. */
  def softmax(logits: Array[Float]): Array[Float] =
    var maxVal = Float.NegativeInfinity
    var i      = 0
    while i < logits.length do
      if logits(i) > maxVal then maxVal = logits(i)
      i += 1
    val out = new Array[Float](logits.length)
    var sum = 0.0f
    i = 0
    while i < logits.length do
      val e = math.exp((logits(i) - maxVal).toDouble).toFloat
      out(i) = e
      sum   += e
      i     += 1
    i = 0
    while i < out.length do
      out(i) /= sum
      i += 1
    out

  /** Nucleus (top-p) filtering on a probability array.
    * Keeps the smallest set of tokens whose cumulative probability ≥ topP,
    * zeroes the rest, then re-normalises so the result still sums to 1.
    * Returns a new array; the input is not modified. */
  def applyTopP(probs: Array[Float], topP: Float): Array[Float] =
    if topP >= 1.0f then return probs.clone()

    val n = probs.length

    // Zero-allocation indirect sort via Long packing:
    // Pack (-prob bits | index) into a Long so java.util.Arrays.sort(long[])
    // sorts by descending probability without boxing a single object.
    // Negative floats in [-1,0] compare correctly as signed ints (larger magnitude
    // = more negative int bits = sorts first when ascending), giving desc-prob order.
    val packed = new Array[Long](n)
    var i = 0
    while i < n do
      val negBits = java.lang.Float.floatToRawIntBits(-probs(i)).toLong
      packed(i) = (negBits << 32) | (i.toLong & 0xFFFFFFFFL)
      i += 1
    java.util.Arrays.sort(packed)  // sorts ascending by negBits → descending by prob

    // Walk until cumulative mass >= topP, mark the cutoff
    var cumProb = 0.0f
    var cutoff  = n
    i = 0
    while i < n do
      val idx = (packed(i) & 0xFFFFFFFFL).toInt
      cumProb += probs(idx)
      if cumProb >= topP && cutoff == n then cutoff = i + 1
      i += 1

    // Build filtered array — keep only the nucleus
    val out = new Array[Float](n)
    i = 0
    while i < cutoff do
      val idx = (packed(i) & 0xFFFFFFFFL).toInt
      out(idx) = probs(idx)
      i += 1

    // Re-normalise
    var sum = 0.0f
    i = 0
    while i < out.length do { sum += out(i); i += 1 }
    if sum > 0.0f then
      i = 0
      while i < out.length do { out(i) /= sum; i += 1 }
    out

  /** Multinomial sample: pick an index with probability proportional to probs.
    * Assumes probs sums to ~1.0. */
  def sampleFromProbs(probs: Array[Float], rng: scala.util.Random): Int =
    val r       = rng.nextFloat()
    var cumProb = 0.0f
    var i       = 0
    while i < probs.length - 1 do
      cumProb += probs(i)
      if r < cumProb then return i
      i += 1
    probs.length - 1  // fallback: last token

  /** Full pipeline: temperature scaling → softmax → top-p nucleus filter → sample.
    *
    * temperature=0.0 short-circuits to [[argmax]] (fully deterministic, no RNG used).
    * topP=1.0 disables nucleus filtering.
    */
  def sample(
      logits:      Array[Float],
      temperature: Float,
      topP:        Float,
      rng:         scala.util.Random
  ): Int =
    if temperature == 0.0f then return argmax(logits)
    val scaled   = applyTemperature(logits.clone(), temperature)
    val probs    = softmax(scaled)
    val filtered = if topP < 1.0f then applyTopP(probs, topP) else probs
    sampleFromProbs(filtered, rng)
