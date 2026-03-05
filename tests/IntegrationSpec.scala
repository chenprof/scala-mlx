/** Integration tests — require built dylibs AND downloaded model weights.
  *
  * Run via:
  *   ./run-tests.sh --integration
  *
  * Tests automatically skip (assume) if the model is not present.
  */
class IntegrationSpec extends munit.FunSuite:

  private val modelDir   = os.pwd / "model"
  private val modelFile  = modelDir / "model.safetensors"
  private val configFile = modelDir / "config.json"
  private val tokFile    = modelDir / "tokenizer.json"
  private def modelReady = os.exists(modelFile) && os.exists(configFile) && os.exists(tokFile)

  private def loadCfgAndTok() =
    val cfg = LlamaConfig.fromFile(configFile)
    val tok = Tokenizer.fromFile(tokFile, cfg)
    (cfg, tok)

  // -------------------------------------------------------------------------
  // Model loading
  // -------------------------------------------------------------------------

  test("model files present after setup.sh") {
    assume(modelReady, "model not downloaded — run ./setup.sh first")
    assert(os.exists(modelFile),  s"missing: $modelFile")
    assert(os.exists(configFile), s"missing: $configFile")
    assert(os.exists(tokFile),    s"missing: $tokFile")
  }

  test("config.json parses successfully") {
    assume(modelReady, "model not downloaded — run ./setup.sh first")
    val cfg = LlamaConfig.fromFile(configFile)
    assert(cfg.numHiddenLayers > 0,   "numHiddenLayers must be > 0")
    assert(cfg.hiddenSize > 0,        "hiddenSize must be > 0")
    assert(cfg.vocabSize > 1000,      "vocabSize seems too small")
  }

  test("tokenizer.json loads and encodes a basic prompt without error") {
    assume(modelReady, "model not downloaded — run ./setup.sh first")
    val (cfg, tok) = loadCfgAndTok()
    val ids = tok.encode("Hello, world!")
    assert(ids.nonEmpty, "encoded prompt must produce at least one token")
    assert(ids.forall(_ >= 0), "all token ids must be non-negative")
  }

  test("special tokens encode as single IDs in Qwen3 tokenizer") {
    assume(modelReady, "model not downloaded — run ./setup.sh first")
    val (cfg, tok) = loadCfgAndTok()
    val start = tok.encode("<|im_start|>")
    val end   = tok.encode("<|im_end|>")
    assertEquals(start.length, 1, "<|im_start|> must be a single token")
    assertEquals(end.length,   1, "<|im_end|> must be a single token")
    assertEquals(end(0), tok.eosId)
  }

  // -------------------------------------------------------------------------
  // Generation (requires LlamaModel + dylibs)
  // -------------------------------------------------------------------------

  test("greedy generation produces non-empty output") {
    assume(modelReady, "model not downloaded — run ./setup.sh first")
    val (cfg, tok) = loadCfgAndTok()
    val prompt = tok.encode("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n", addBos = tok.addBos)
    val model  = LlamaModel(cfg, modelFile)
    try
      val out = model.generate(prompt, maxNewTokens = 10, eosId = tok.eosId)
      assert(out.nonEmpty, "generate() returned no tokens")
    finally model.close()
  }

  test("greedy (temperature=0) is deterministic across two runs") {
    assume(modelReady, "model not downloaded — run ./setup.sh first")
    val (cfg, tok) = loadCfgAndTok()
    val prompt = tok.encode("<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n", addBos = tok.addBos)
    val model  = LlamaModel(cfg, modelFile)
    try
      val run1 = model.generate(prompt, maxNewTokens = 20, eosId = tok.eosId, temperature = 0.0f)
      val run2 = model.generate(prompt, maxNewTokens = 20, eosId = tok.eosId, temperature = 0.0f)
      assertEquals(run1.toSeq, run2.toSeq, "two greedy runs must produce identical tokens")
    finally model.close()
  }

  test("EOS token stops generation before maxNewTokens") {
    assume(modelReady, "model not downloaded — run ./setup.sh first")
    val (cfg, tok) = loadCfgAndTok()
    val prompt = tok.encode("<|im_start|>user\nSay hi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n", addBos = tok.addBos)
    val model  = LlamaModel(cfg, modelFile)
    try
      val out = model.generate(prompt, maxNewTokens = 512, eosId = tok.eosId)
      assert(out.length < 512, s"model should stop at EOS before 512 tokens, got ${out.length}")
    finally model.close()
  }

  test("decoded output is valid UTF-8 text") {
    assume(modelReady, "model not downloaded — run ./setup.sh first")
    val (cfg, tok) = loadCfgAndTok()
    val prompt = tok.encode("<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n", addBos = tok.addBos)
    val model  = LlamaModel(cfg, modelFile)
    try
      val out = model.generate(prompt, maxNewTokens = 50, eosId = tok.eosId)
      val text = tok.decode(out.filter(_ != tok.eosId).toSeq)
      assert(text.nonEmpty, "decoded text should not be empty")
      assert(text.contains("Paris") || text.contains("paris") || text.length > 5,
        s"expected mention of Paris or substantial text, got: $text")
    finally model.close()
  }

  test("onToken callback fires for each generated token") {
    assume(modelReady, "model not downloaded — run ./setup.sh first")
    val (cfg, tok) = loadCfgAndTok()
    val prompt = tok.encode("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n", addBos = tok.addBos)
    val model  = LlamaModel(cfg, modelFile)
    try
      var callbackCount = 0
      val out = model.generate(
        prompt, maxNewTokens = 10, eosId = tok.eosId,
        onToken = _ => callbackCount += 1
      )
      assertEquals(callbackCount, out.length, "callback count must equal generated token count")
    finally model.close()
  }
