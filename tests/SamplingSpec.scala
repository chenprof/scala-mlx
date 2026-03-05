/** Unit tests for Sampling — pure math, no model, no FFI, no file I/O. */
class SamplingSpec extends munit.FunSuite:

  // -------------------------------------------------------------------------
  // argmax
  // -------------------------------------------------------------------------

  test("argmax: returns index of maximum logit") {
    val logits = Array(0.1f, 5.0f, 0.3f, 2.0f)
    assertEquals(Sampling.argmax(logits), 1)
  }

  test("argmax: handles all-negative logits") {
    val logits = Array(-3.0f, -1.0f, -2.0f)
    assertEquals(Sampling.argmax(logits), 1)
  }

  test("argmax: single element always returns index 0") {
    assertEquals(Sampling.argmax(Array(42.0f)), 0)
  }

  test("argmax: tie — returns a valid index in range") {
    val logits = Array(1.0f, 1.0f, 1.0f)
    val idx = Sampling.argmax(logits)
    assert(idx >= 0 && idx < logits.length, s"argmax returned out-of-range index $idx")
  }

  test("argmax: last element wins") {
    val logits = Array(0.0f, 0.0f, 0.0f, 9.9f)
    assertEquals(Sampling.argmax(logits), 3)
  }

  // -------------------------------------------------------------------------
  // softmax
  // -------------------------------------------------------------------------

  test("softmax: probabilities sum to 1.0") {
    val logits = Array(1.0f, 2.0f, 3.0f, 0.5f)
    val probs = Sampling.softmax(logits)
    assertEqualsDouble(probs.sum.toDouble, 1.0, 1e-5)
  }

  test("softmax: all probabilities are non-negative") {
    val logits = Array(-100.0f, 100.0f, 0.0f)
    Sampling.softmax(logits).foreach(p => assert(p >= 0.0f, s"negative probability $p"))
  }

  test("softmax: higher logit gets higher probability") {
    val logits = Array(1.0f, 3.0f, 2.0f)
    val probs = Sampling.softmax(logits)
    assert(probs(1) > probs(2), "p(1) should be > p(2)")
    assert(probs(2) > probs(0), "p(2) should be > p(0)")
  }

  test("softmax: uniform logits yield uniform probabilities") {
    val logits = Array(0.0f, 0.0f, 0.0f, 0.0f)
    val probs = Sampling.softmax(logits)
    probs.foreach(p => assertEqualsDouble(p.toDouble, 0.25, 1e-5))
  }

  test("softmax: numerically stable against large logits") {
    // Without max subtraction this would overflow to NaN
    val logits = Array(1000.0f, 1001.0f, 999.0f)
    val probs = Sampling.softmax(logits)
    assert(!probs.exists(_.isNaN), "softmax produced NaN for large logits")
    assertEqualsDouble(probs.sum.toDouble, 1.0, 1e-4)
  }

  // -------------------------------------------------------------------------
  // applyTemperature
  // -------------------------------------------------------------------------

  test("applyTemperature: divides all logits by temperature") {
    val logits = Array(2.0f, 4.0f, 6.0f)
    val scaled = Sampling.applyTemperature(logits.clone(), 2.0f)
    assertEqualsDouble(scaled(0).toDouble, 1.0, 1e-5)
    assertEqualsDouble(scaled(1).toDouble, 2.0, 1e-5)
    assertEqualsDouble(scaled(2).toDouble, 3.0, 1e-5)
  }

  test("applyTemperature: temperature=1.0 is identity") {
    val logits = Array(1.0f, 2.0f, 3.0f)
    val scaled = Sampling.applyTemperature(logits.clone(), 1.0f)
    logits.zip(scaled).foreach { case (orig, s) =>
      assertEqualsDouble(s.toDouble, orig.toDouble, 1e-5)
    }
  }

  test("applyTemperature: high temperature flattens relative differences") {
    val logits = Array(1.0f, 10.0f)
    val highTemp = Sampling.softmax(Sampling.applyTemperature(logits.clone(), 10.0f))
    val lowTemp  = Sampling.softmax(Sampling.applyTemperature(logits.clone(), 0.1f))
    // High temperature: p(0)/p(1) closer to 1; low temperature: p(1) dominates
    val highRatio = highTemp(1) / highTemp(0)
    val lowRatio  = lowTemp(1)  / lowTemp(0)
    assert(lowRatio > highRatio, "low temperature should sharpen the distribution more")
  }

  // -------------------------------------------------------------------------
  // applyTopP
  // -------------------------------------------------------------------------

  test("applyTopP: topP=1.0 retains all probability mass") {
    val probs = Array(0.1f, 0.4f, 0.3f, 0.2f)
    val filtered = Sampling.applyTopP(probs.clone(), 1.0f)
    assertEqualsDouble(filtered.sum.toDouble, 1.0, 1e-4)
    filtered.foreach(p => assert(p >= 0.0f))
  }

  test("applyTopP: very small topP keeps only the dominant token") {
    // Token 0 has 90% probability — only it should survive at topP=0.01
    val probs = Array(0.9f, 0.033f, 0.034f, 0.033f)
    val filtered = Sampling.applyTopP(probs.clone(), 0.01f)
    assertEquals(filtered.count(_ > 0.0f), 1)
    // The surviving token should hold all the probability mass
    assertEqualsDouble(filtered.max.toDouble, 1.0, 1e-4)
  }

  test("applyTopP: nucleus covers at least topP of probability mass") {
    val probs = Array(0.5f, 0.3f, 0.1f, 0.1f)
    val filtered = Sampling.applyTopP(probs.clone(), 0.8f)
    val coveredMass = filtered.filter(_ > 0.0f).sum
    assert(coveredMass >= 0.79f, s"covered mass $coveredMass < 0.79")
  }

  test("applyTopP: result is a valid probability distribution") {
    val probs = Array(0.25f, 0.25f, 0.25f, 0.25f)
    val filtered = Sampling.applyTopP(probs.clone(), 0.5f)
    assertEqualsDouble(filtered.sum.toDouble, 1.0, 1e-4)
    filtered.foreach(p => assert(p >= 0.0f, s"negative probability $p after top-p"))
  }

  // -------------------------------------------------------------------------
  // sample (full pipeline)
  // -------------------------------------------------------------------------

  test("sample: temperature=0.0 always returns argmax") {
    val logits = Array(0.1f, 8.0f, 0.5f, 1.0f)
    val rng = scala.util.Random(42L)
    val results = (0 until 20).map(_ => Sampling.sample(logits, 0.0f, 1.0f, rng))
    assert(results.forall(_ == 1), s"expected all 1, got: $results")
  }

  test("sample: result is always a valid vocab index") {
    val logits = Array(1.0f, 2.0f, 3.0f, 1.5f, 0.5f)
    val rng = scala.util.Random(123L)
    for _ <- 0 until 50 do
      val idx = Sampling.sample(logits, 1.0f, 1.0f, rng)
      assert(idx >= 0 && idx < logits.length, s"out-of-range index: $idx")
  }

  test("sample: heavily biased logits → dominant token sampled most often") {
    // Token 2 has logit 10.0, rest near 0 — should win >95% of the time
    val logits = Array(0.0f, 0.0f, 10.0f, 0.0f)
    val rng = scala.util.Random(42L)
    val counts = Array.fill(4)(0)
    for _ <- 0 until 100 do
      counts(Sampling.sample(logits, 1.0f, 1.0f, rng)) += 1
    assert(counts(2) > 90, s"expected token 2 >90/100, got ${counts(2)}")
  }

  test("sample: different RNG seeds produce different token sequences") {
    val logits = Array.fill(8)(1.0f) // uniform → random choices
    val rng1 = scala.util.Random(1L)
    val rng2 = scala.util.Random(9999L)
    val s1 = (0 until 20).map(_ => Sampling.sample(logits, 1.0f, 1.0f, rng1))
    val s2 = (0 until 20).map(_ => Sampling.sample(logits, 1.0f, 1.0f, rng2))
    assert(s1 != s2, "different seeds produced identical 20-token sequences")
  }

  test("sample: topP=0.01 forces selection to argmax (single survivor)") {
    // After top-p at 0.01 only the max-prob token survives → deterministic
    val logits = Array(0.0f, 0.0f, 10.0f, 0.0f) // token 2 dominates
    val rng = scala.util.Random(77L)
    val results = (0 until 20).map(_ => Sampling.sample(logits, 1.0f, 0.01f, rng))
    assert(results.forall(_ == 2), s"expected all 2 with tight topP, got: $results")
  }
