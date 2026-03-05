/** Unit tests for LlamaConfig — JSON parsing, no model weights, no FFI. */
class LlamaConfigSpec extends munit.FunSuite:

  // -------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------

  private def writeConfig(json: String): os.Path =
    val d = os.temp.dir()
    os.write(d / "config.json", json)
    d / "config.json"

  // Realistic Qwen3-0.6B config (all fields present)
  private val qwen3Json =
    """{
      |  "model_type": "qwen3",
      |  "num_hidden_layers": 28,
      |  "hidden_size": 1024,
      |  "num_attention_heads": 16,
      |  "num_key_value_heads": 8,
      |  "head_dim": 128,
      |  "intermediate_size": 3072,
      |  "vocab_size": 151936,
      |  "rms_norm_eps": 1e-6,
      |  "rope_theta": 1000000.0,
      |  "bos_token_id": null,
      |  "eos_token_id": 151645
      |}""".stripMargin

  // -------------------------------------------------------------------------
  // Field parsing
  // -------------------------------------------------------------------------

  test("parses num_hidden_layers") {
    assertEquals(LlamaConfig.fromFile(writeConfig(qwen3Json)).numHiddenLayers, 28)
  }

  test("parses hidden_size") {
    assertEquals(LlamaConfig.fromFile(writeConfig(qwen3Json)).hiddenSize, 1024)
  }

  test("parses num_attention_heads") {
    assertEquals(LlamaConfig.fromFile(writeConfig(qwen3Json)).numAttentionHeads, 16)
  }

  test("parses num_key_value_heads") {
    assertEquals(LlamaConfig.fromFile(writeConfig(qwen3Json)).numKeyValueHeads, 8)
  }

  test("parses intermediate_size") {
    assertEquals(LlamaConfig.fromFile(writeConfig(qwen3Json)).intermediateSize, 3072)
  }

  test("parses vocab_size") {
    assertEquals(LlamaConfig.fromFile(writeConfig(qwen3Json)).vocabSize, 151936)
  }

  test("parses rms_norm_eps") {
    assertEqualsDouble(LlamaConfig.fromFile(writeConfig(qwen3Json)).rmsNormEps.toDouble, 1e-6, 1e-10)
  }

  test("parses rope_theta") {
    assertEqualsDouble(LlamaConfig.fromFile(writeConfig(qwen3Json)).ropeTheta.toDouble, 1000000.0, 1.0)
  }

  test("parses eos_token_id") {
    assertEquals(LlamaConfig.fromFile(writeConfig(qwen3Json)).eosTokenId, 151645)
  }

  // -------------------------------------------------------------------------
  // head_dim: explicit vs derived
  // -------------------------------------------------------------------------

  test("uses explicit head_dim when present (Qwen3 style)") {
    assertEquals(LlamaConfig.fromFile(writeConfig(qwen3Json)).headDim, 128)
  }

  test("derives head_dim from hidden_size / num_attention_heads when absent") {
    // Remove "head_dim": 128, — preceding comma stays on num_key_value_heads line
    val json = qwen3Json.replace("""  "head_dim": 128,""" + "\n", "")
    val cfg = LlamaConfig.fromFile(writeConfig(json))
    assertEquals(cfg.headDim, 1024 / 16) // 64
  }

  test("derived head_dim: LLaMA-style 32 heads × 128 = 4096 hidden") {
    val json =
      """{
        |  "num_hidden_layers": 32,
        |  "hidden_size": 4096,
        |  "num_attention_heads": 32,
        |  "num_key_value_heads": 32,
        |  "intermediate_size": 11008,
        |  "vocab_size": 32000,
        |  "rms_norm_eps": 1e-5,
        |  "bos_token_id": 1,
        |  "eos_token_id": 2
        |}""".stripMargin
    val cfg = LlamaConfig.fromFile(writeConfig(json))
    assertEquals(cfg.headDim, 4096 / 32) // 128
  }

  // -------------------------------------------------------------------------
  // bos_token_id: null vs integer
  // -------------------------------------------------------------------------

  test("bos_token_id null → bosTokenId = None (Qwen3)") {
    assertEquals(LlamaConfig.fromFile(writeConfig(qwen3Json)).bosTokenId, None)
  }

  test("bos_token_id integer → bosTokenId = Some(id)") {
    val json = qwen3Json.replace("\"bos_token_id\": null", "\"bos_token_id\": 1")
    assertEquals(LlamaConfig.fromFile(writeConfig(json)).bosTokenId, Some(1))
  }

  // -------------------------------------------------------------------------
  // rope_theta: default fallback
  // -------------------------------------------------------------------------

  test("rope_theta defaults to 10000.0 when absent") {
    val json = qwen3Json.replace("""  "rope_theta": 1000000.0,""" + "\n", "")
    assertEqualsDouble(LlamaConfig.fromFile(writeConfig(json)).ropeTheta.toDouble, 10000.0, 1.0)
  }

  // -------------------------------------------------------------------------
  // Derived fields
  // -------------------------------------------------------------------------

  test("ropeDims equals headDim") {
    val cfg = LlamaConfig.fromFile(writeConfig(qwen3Json))
    assertEquals(cfg.ropeDims, cfg.headDim)
  }

  // -------------------------------------------------------------------------
  // Sanity: actual Qwen3-0.6B values
  // -------------------------------------------------------------------------

  test("Qwen3-0.6B: all parsed fields match known architecture spec") {
    val cfg = LlamaConfig.fromFile(writeConfig(qwen3Json))
    assertEquals(cfg.numHiddenLayers, 28)
    assertEquals(cfg.hiddenSize, 1024)
    assertEquals(cfg.numAttentionHeads, 16)
    assertEquals(cfg.numKeyValueHeads, 8)
    assertEquals(cfg.headDim, 128)   // explicitly set, != hiddenSize/numHeads (64)
    assertEquals(cfg.vocabSize, 151936)
    assertEquals(cfg.bosTokenId, None)
    assertEquals(cfg.eosTokenId, 151645)
  }
