/** Unit tests for Tokenizer — no model weights, no FFI.
  *
  * Uses a minimal hand-crafted tokenizer.json written to a temp directory.
  * The test vocab (Metaspace mode) is:
  *
  *   "▁"→0, "a"→1, "b"→2, "c"→3
  *   "▁a"→10, "▁ab"→11, "▁abc"→12
  *
  * Merges (in priority order):
  *   0: ("▁","a") → "▁a"
  *   1: ("▁a","b") → "▁ab"
  *   2: ("▁ab","c") → "▁abc"
  *
  * Added tokens: <|im_end|>→101, <think>→102, </think>→103
  *
  * BPE trace for "abc":
  *   splitCodePoints("▁abc") = ["▁","a","b","c"]
  *   merge0 → ["▁a","b","c"]
  *   merge1 → ["▁ab","c"]
  *   merge2 → ["▁abc"] → vocab lookup → 12
  */
class TokenizerSpec extends munit.FunSuite:

  // -------------------------------------------------------------------------
  // byte2unicode / unicode2byte — pure table tests, no file I/O
  // -------------------------------------------------------------------------

  test("byte2unicode: space (0x20) maps to Ġ (U+0120)") {
    assertEquals(Tokenizer.byte2unicode(32), '\u0120')
  }

  test("byte2unicode: ASCII printable 'A' (0x41) is identity") {
    assertEquals(Tokenizer.byte2unicode(65), 'A')
  }

  test("byte2unicode: newline (0x0A) maps to Ċ (U+010A)") {
    assertEquals(Tokenizer.byte2unicode(10), '\u010A')
  }

  test("byte2unicode: covers all 256 byte values") {
    assertEquals(Tokenizer.byte2unicode.size, 256)
  }

  test("byte2unicode / unicode2byte are inverse for all 256 bytes") {
    for b <- 0 until 256 do
      val c = Tokenizer.byte2unicode(b)
      assertEquals(
        Tokenizer.unicode2byte(c), b,
        s"roundtrip failed for byte $b → char '$c' → byte ${Tokenizer.unicode2byte.getOrElse(c, -1)}"
      )
  }

  test("unicode2byte: all mapped chars round-trip back to their byte") {
    for (c, b) <- Tokenizer.unicode2byte do
      assertEquals(
        Tokenizer.byte2unicode(b), c,
        s"roundtrip failed for char '$c' → byte $b → char '${Tokenizer.byte2unicode(b)}'"
      )
  }

  // -------------------------------------------------------------------------
  // Tokenizer backed by a minimal hand-crafted tokenizer.json
  // -------------------------------------------------------------------------

  // Metaspace tokenizer JSON.  ▁ = \u2581.
  // Merges use the array format (Qwen3 style) to exercise that branch.
  private val tokJson =
    """{
      |  "model": {
      |    "type": "BPE",
      |    "vocab": {
      |      "\u2581": 0,
      |      "a": 1,
      |      "b": 2,
      |      "c": 3,
      |      "\u2581a":   10,
      |      "\u2581ab":  11,
      |      "\u2581abc": 12
      |    },
      |    "merges": [
      |      ["\u2581",   "a"  ],
      |      ["\u2581a",  "b"  ],
      |      ["\u2581ab", "c"  ]
      |    ]
      |  },
      |  "pre_tokenizer": {"type": "Metaspace"},
      |  "added_tokens": [
      |    {"id": 101, "content": "<|im_end|>"},
      |    {"id": 102, "content": "<think>"},
      |    {"id": 103, "content": "</think>"}
      |  ]
      |}""".stripMargin

  private val dummyCfg = LlamaConfig(
    numHiddenLayers   = 1,
    hiddenSize        = 64,
    numAttentionHeads = 2,
    numKeyValueHeads  = 2,
    headDim           = 32,
    intermediateSize  = 128,
    vocabSize         = 200,
    rmsNormEps        = 1e-5f,
    ropeTheta         = 10000.0f,
    bosTokenId        = None,
    eosTokenId        = 101
  )

  private lazy val tmpDir: os.Path =
    val d = os.temp.dir()
    os.write(d / "tokenizer.json", tokJson)
    d

  private lazy val tok: Tokenizer =
    Tokenizer.fromFile(tmpDir / "tokenizer.json", dummyCfg)

  // -------------------------------------------------------------------------
  // BPE merge behaviour
  // -------------------------------------------------------------------------

  test("BPE: 'abc' fully merges to single token 12") {
    assertEquals(tok.encode("abc").toSeq, Seq(12))
  }

  test("BPE: 'ab' stops after two merges → token 11") {
    assertEquals(tok.encode("ab").toSeq, Seq(11))
  }

  test("BPE: 'a' stops after first merge → token 10") {
    assertEquals(tok.encode("a").toSeq, Seq(10))
  }

  test("BPE: decode of fully merged token round-trips to original text") {
    assertEquals(tok.decode(Seq(12)), "abc")
  }

  test("BPE: decode of partial merge round-trips correctly") {
    assertEquals(tok.decode(Seq(11)), "ab")
  }

  // -------------------------------------------------------------------------
  // Special token encoding (added_tokens bypass BPE entirely)
  // -------------------------------------------------------------------------

  test("special: <|im_end|> encodes as single ID 101") {
    assertEquals(tok.encode("<|im_end|>").toSeq, Seq(101))
  }

  test("special: <think> encodes as single ID 102") {
    assertEquals(tok.encode("<think>").toSeq, Seq(102))
  }

  test("special: </think> encodes as single ID 103") {
    assertEquals(tok.encode("</think>").toSeq, Seq(103))
  }

  test("special: two consecutive special tokens encode correctly") {
    assertEquals(tok.encode("<think></think>").toSeq, Seq(102, 103))
  }

  test("special: special token surrounded by plain text — [102, 12, 103]") {
    assertEquals(tok.encode("<think>abc</think>").toSeq, Seq(102, 12, 103))
  }

  test("special: plain text followed by special token — [12, 101]") {
    assertEquals(tok.encode("abc<|im_end|>").toSeq, Seq(12, 101))
  }

  test("special: special token is NOT split by BPE (single token, not chars)") {
    val ids = tok.encode("<think>")
    assertEquals(ids.length, 1)
    assertEquals(ids(0), 102)
  }

  test("special: longest match wins — </think> preferred over shorter prefixes") {
    // "</think>" (8 chars) must beat any shorter special token that could start with '<'
    val ids = tok.encode("</think>")
    assertEquals(ids.length, 1)
    assertEquals(ids(0), 103)
  }

  // -------------------------------------------------------------------------
  // EOS / BOS
  // -------------------------------------------------------------------------

  test("eosId: taken from added_tokens <|im_end|>=101") {
    assertEquals(tok.eosId, 101)
  }

  test("bosId: None when config.bosTokenId is None") {
    assertEquals(tok.bosId, None)
  }

  // Tokenizer.json with TemplateProcessing post_processor that references BOS
  private val tokJsonWithBos =
    """{
      |  "model": {
      |    "type": "BPE",
      |    "vocab": {
      |      "\u2581": 0, "a": 1, "b": 2, "c": 3,
      |      "\u2581a": 10, "\u2581ab": 11, "\u2581abc": 12
      |    },
      |    "merges": [["\u2581","a"], ["\u2581a","b"], ["\u2581ab","c"]]
      |  },
      |  "pre_tokenizer": {"type": "Metaspace"},
      |  "added_tokens": [],
      |  "post_processor": {
      |    "type": "TemplateProcessing",
      |    "single": [
      |      {"SpecialToken": {"id": "<bos>", "type_id": 0}},
      |      {"Sequence": {"id": "A", "type_id": 0}}
      |    ]
      |  }
      |}""".stripMargin

  test("encode with addBos=true prepends BOS when post_processor references BOS") {
    val d = os.temp.dir()
    os.write(d / "tokenizer.json", tokJsonWithBos)
    val cfgWithBos = dummyCfg.copy(bosTokenId = Some(99))
    val t = Tokenizer.fromFile(d / "tokenizer.json", cfgWithBos)
    val ids = t.encode("abc", addBos = true)
    assertEquals(ids(0), 99)  // BOS
    assertEquals(ids(1), 12)  // "abc" token
    assertEquals(ids.length, 2)
  }

  test("encode with addBos=false does NOT prepend BOS even when bosId is defined") {
    val d = os.temp.dir()
    os.write(d / "tokenizer.json", tokJsonWithBos)
    val cfgWithBos = dummyCfg.copy(bosTokenId = Some(99))
    val t = Tokenizer.fromFile(d / "tokenizer.json", cfgWithBos)
    val ids = t.encode("abc", addBos = false)
    assertEquals(ids(0), 12)
    assertEquals(ids.length, 1)
  }

  // -------------------------------------------------------------------------
  // Tokenizer mode detection
  // -------------------------------------------------------------------------

  test("mode: Metaspace detected from pre_tokenizer.type=Metaspace JSON") {
    // Encoding with Metaspace prefixes ▁ — token 10 = ▁a, confirming Metaspace mode
    assertEquals(tok.encode("a").toSeq, Seq(10))
  }

  test("mode: ByteLevel detected from pre_tokenizer.type=ByteLevel JSON") {
    val byteLevelJson =
      """{
        |  "model": {
        |    "type": "BPE",
        |    "vocab": {"A": 65, "B": 66},
        |    "merges": []
        |  },
        |  "pre_tokenizer": {"type": "ByteLevel"},
        |  "added_tokens": []
        |}""".stripMargin
    val d = os.temp.dir()
    os.write(d / "tokenizer.json", byteLevelJson)
    val t = Tokenizer.fromFile(d / "tokenizer.json", dummyCfg)
    // In ByteLevel mode, 'A' (0x41=65) stays 'A' — vocab has "A"→65
    val ids = t.encode("A")
    assertEquals(ids(0), 65)
  }

  // -------------------------------------------------------------------------
  // String merge format (Qwen3 array-style vs LLaMA string-style)
  // -------------------------------------------------------------------------

  test("merges: string-format merges (LLaMA style '▁ a') are parsed correctly") {
    val llamaJson =
      s"""{
        |  "model": {
        |    "type": "BPE",
        |    "vocab": {
        |      "\u2581": 0, "a": 1, "\u2581a": 10
        |    },
        |    "merges": ["\u2581 a"]
        |  },
        |  "pre_tokenizer": {"type": "Metaspace"},
        |  "added_tokens": []
        |}""".stripMargin
    val d = os.temp.dir()
    os.write(d / "tokenizer.json", llamaJson)
    val t = Tokenizer.fromFile(d / "tokenizer.json", dummyCfg)
    assertEquals(t.encode("a").toSeq, Seq(10))
  }
