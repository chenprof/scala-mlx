import ujson.*

/** BPE tokenizer loaded from a HuggingFace tokenizer.json.
  *
  * Supports two pre-tokenizer modes, detected automatically from the JSON:
  *
  *   Metaspace  — LLaMA / TinyLlama (U+2581 ▁ replaces spaces)
  *   ByteLevel  — Qwen3 / GPT-2 style (each byte mapped to a Unicode char)
  *
  * The BPE merge algorithm is identical in both cases; only the
  * pre-tokenization and decode steps differ.
  */
class Tokenizer private (
    vocab:    Map[String, Int],    // BPE vocab + added_tokens merged
    merges:   Map[(String, String), Int],
    mode:     Tokenizer.Mode,
    val bosId: Option[Int],
    val eosId: Int,
    val addBos: Boolean,
    specialTokens: Map[String, Int],  // added_tokens only (for pre-tokenization split)
    splitPattern: Option[java.util.regex.Pattern]  // regex for pre-tokenization (from tokenizer.json)
):
  private val idToToken: Map[Int, String] = vocab.map(_.swap)
  private val unkId: Int = vocab.getOrElse("<unk>", 0)
  // Sorted by length descending so longer tokens are matched first
  private val specialList: Array[(String, Int)] =
    specialTokens.toArray.sortBy(-_._1.length)

  /* ----------------------------------------------------------------
   * Encode
   * ---------------------------------------------------------------- */

  def encode(text: String, addBos: Boolean = false): Array[Int] =
    // Step 1: split on special tokens (added_tokens) — they bypass BPE entirely
    val chunks: Seq[Either[Int, String]] = splitSpecial(text)

    // Step 2: BPE-encode the plain-text chunks; special-token chunks are already IDs
    val ids = chunks.flatMap {
      case Left(id)   => Seq(id)
      case Right(str) => encodePlain(str)
    }
    bosId match
      case Some(id) if addBos => id +: ids.toArray
      case _                  => ids.toArray

  /** Split `text` at special-token boundaries.
    * Left(id) = a special token;  Right(s) = a plain-text fragment. */
  private def splitSpecial(text: String): Seq[Either[Int, String]] =
    val result = scala.collection.mutable.ArrayBuffer.empty[Either[Int, String]]
    var pos = 0
    while pos < text.length do
      // Try to match any special token at the current position (longest first)
      var matched = false
      var k = 0
      while k < specialList.length && !matched do
        val entry = specialList(k)
        val tok: String = entry._1
        val id: Int = entry._2
        if text.startsWith(tok, pos) then
          result += Left(id)
          pos += tok.length
          matched = true
        k += 1
      if !matched then
        // Find the start of the next special token (or end of string)
        var end = pos + 1
        var found = false
        while end < text.length && !found do
          var j = 0
          while j < specialList.length && !found do
            if text.startsWith(specialList(j)._1, end) then found = true
            j += 1
          if !found then end += 1
        result += Right(text.substring(pos, end))
        pos = end
    result.toSeq

  /** BPE-encode a plain-text string (no special tokens). */
  private def encodePlain(str: String): Seq[Int] =
    val preTokens: Seq[String] = mode match
      case Tokenizer.Mode.Metaspace =>
        Seq(("\u2581" + str).replace(" ", "\u2581"))

      case Tokenizer.Mode.ByteLevel =>
        // Use the regex split pattern from tokenizer.json if available,
        // otherwise fall back to whitespace-boundary split.
        val chunks: Seq[String] = splitPattern match
          case Some(pat) =>
            val m = pat.matcher(str)
            val buf = scala.collection.mutable.ArrayBuffer.empty[String]
            while m.find() do buf += m.group()
            buf.toSeq
          case None =>
            val buf = scala.collection.mutable.ArrayBuffer.empty[String]
            val sb  = new java.lang.StringBuilder
            var i   = 0
            while i < str.length do
              val c        = str.charAt(i)
              val isWs     = c == ' ' || c == '\n' || c == '\t' || c == '\r'
              val prevIsWs = sb.length > 0 && {
                val p = sb.charAt(sb.length - 1)
                p == ' ' || p == '\n' || p == '\t' || p == '\r'
              }
              if sb.length > 0 && isWs != prevIsWs then
                buf += sb.toString; sb.setLength(0)
              sb.append(c); i += 1
            if sb.length > 0 then buf += sb.toString
            buf.toSeq
        chunks.filter(_.nonEmpty)
          .map(p => p.getBytes("UTF-8").map(b => Tokenizer.byte2unicode(b & 0xFF)).mkString)

    preTokens.flatMap(bpe).map(t => vocab.getOrElse(t, unkId))

  /** Split a string into individual Unicode code-point strings.
    * Avoids java.util.stream.IntStream which is not available in Scala Native. */
  private def splitCodePoints(s: String): Array[String] =
    val buf = scala.collection.mutable.ArrayBuffer.empty[String]
    var i = 0
    while i < s.length do
      val cp = s.codePointAt(i)
      buf += new String(Character.toChars(cp))
      i += Character.charCount(cp)
    buf.toArray

  /** BPE: character-split a pre-token, then greedily apply merge rules. */
  private def bpe(word: String): Array[String] =
    var toks: Array[String] = splitCodePoints(word)
    var changed = true
    while changed do
      changed = false
      var bestPriority = Int.MaxValue
      var bestPos      = -1
      var i = 0
      while i < toks.length - 1 do
        merges.get((toks(i), toks(i + 1))) match
          case Some(p) if p < bestPriority =>
            bestPriority = p; bestPos = i
          case _ =>
        i += 1
      if bestPos >= 0 then
        val merged = toks(bestPos) + toks(bestPos + 1)
        toks = toks.take(bestPos) ++ Array(merged) ++ toks.drop(bestPos + 2)
        changed = true
    toks

  /* ----------------------------------------------------------------
   * Decode
   * ---------------------------------------------------------------- */

  def decode(ids: Seq[Int]): String =
    mode match
      case Tokenizer.Mode.Metaspace =>
        ids.map(decodeToken).mkString.replace("\u2581", " ").dropWhile(_ == ' ')

      case Tokenizer.Mode.ByteLevel =>
        val charStr = ids.map(id => idToToken.getOrElse(id, "")).mkString
        val bytes   = charStr.flatMap(c => Tokenizer.unicode2byte.get(c).map(_.toByte)).toArray
        new String(bytes, "UTF-8")

  def decodeToken(id: Int): String =
    val tok = idToToken.getOrElse(id, "")
    mode match
      case Tokenizer.Mode.ByteLevel =>
        val bytes = tok.flatMap(c => Tokenizer.unicode2byte.get(c).map(_.toByte)).toArray
        new String(bytes, "UTF-8")
      case Tokenizer.Mode.Metaspace =>
        if tok.startsWith("<0x") && tok.endsWith(">") then
          try new String(Array(Integer.parseInt(tok.slice(3, tok.length - 1), 16).toByte), "ISO-8859-1")
          catch case _: Exception => tok
        else tok


object Tokenizer:

  enum Mode { case Metaspace, ByteLevel }

  /* ----------------------------------------------------------------
   * GPT-2 byte ↔ unicode table (used by ByteLevel tokenizers)
   * Maps every byte value 0-255 to a printable Unicode character.
   * Space (0x20) → Ġ (U+0120), newline (0x0A) → Ċ (U+010A), etc.
   * ---------------------------------------------------------------- */
  val byte2unicode: Map[Int, Char] =
    val initial = ((33 to 126) ++ (161 to 172) ++ (174 to 255)).toSet
    val bs = scala.collection.mutable.ArrayBuffer.from(initial.toSeq.sorted)
    val cs = scala.collection.mutable.ArrayBuffer.from(initial.toSeq.sorted)
    var n = 0
    for b <- 0 until 256 do
      if !initial.contains(b) then
        bs += b
        cs += (256 + n)
        n  += 1
    bs.zip(cs).map { case (b, c) => b -> c.toChar }.toMap

  val unicode2byte: Map[Char, Int] = byte2unicode.map(_.swap)

  /* ----------------------------------------------------------------
   * Load from tokenizer.json
   * ---------------------------------------------------------------- */

  def fromFile(path: os.Path, cfg: LlamaConfig): Tokenizer =
    val json      = ujson.read(os.read(path))
    val modelJson = json("model")

    val bpeVocab: Map[String, Int] =
      modelJson("vocab").obj.toMap.map { case (k, v) => k -> v.num.toInt }

    val merges: Map[(String, String), Int] =
      modelJson("merges").arr.zipWithIndex.map { case (v, i) =>
        val pair = v match
          case ujson.Arr(parts) => (parts(0).str, parts(1).str)
          case _                => val p = v.str.split(" ", 2); (p(0), p(1))
        pair -> i
      }.toMap

    // Load added_tokens (special tokens like <|im_start|>, <|im_end|>, <think>, …)
    val specialTokens: Map[String, Int] =
      json.obj.get("added_tokens").map(
        _.arr.map(t => t("content").str -> t("id").num.toInt).toMap
      ).getOrElse(Map.empty)

    // Merge added_tokens into the vocab so decoding and encoding both work
    val vocab = bpeVocab ++ specialTokens

    // Detect pre-tokenizer mode from the tokenizer.json
    val mode = json.obj.get("pre_tokenizer") match
      case Some(pt) if pt.obj.get("type").exists(_.str == "ByteLevel") => Mode.ByteLevel
      case Some(pt) if pt.obj.get("type").exists(_.str == "Sequence")  =>
        val hasByteLevel = pt.obj.get("pretokenizers")
          .map(_.arr.exists(_.obj.get("type").exists(_.str == "ByteLevel")))
          .getOrElse(false)
        if hasByteLevel then Mode.ByteLevel else Mode.Metaspace
      case _ => Mode.Metaspace

    // EOS: prefer <|im_end|> from added_tokens over config.json
    val eosId = specialTokens.getOrElse("<|im_end|>",
                  specialTokens.getOrElse("</s>", cfg.eosTokenId))

    // BOS: only add if post_processor is TemplateProcessing that references BOS.
    // Qwen3/GPT-style models use ByteLevel post_processor → no BOS.
    val addBos: Boolean = json.obj.get("post_processor") match
      case Some(pp) if pp.obj.get("type").exists(_.str == "TemplateProcessing") =>
        // Has a template post-processor — check if it references a BOS-like token
        pp.obj.get("single").exists(_.arr.exists(item =>
          item.obj.get("SpecialToken").exists(st =>
            st.obj.get("id").exists(_.str.contains("bos"))
          )
        ))
      case _ => false
    val bosId = if addBos then cfg.bosTokenId else None

    // Extract pre-tokenization regex from Split pre-tokenizer (GPT-4/Qwen3 style).
    // Remove negative lookahead (?!\S) which Scala Native's RE2 doesn't support.
    val splitPattern: Option[java.util.regex.Pattern] =
      val splitRegex = json.obj.get("pre_tokenizer").flatMap { pt =>
        pt.obj.get("type").map(_.str) match
          case Some("Split") =>
            pt.obj.get("pattern").flatMap(_.obj.get("Regex").map(_.str))
          case Some("Sequence") =>
            pt.obj.get("pretokenizers").flatMap(_.arr.collectFirst {
              case p if p.obj.get("type").exists(_.str == "Split") =>
                p.obj.get("pattern").flatMap(_.obj.get("Regex").map(_.str))
            }.flatten)
          case _ => None
      }
      splitRegex.map { r =>
        // Remove negative lookahead (unsupported in RE2): \s+(?!\S)|\s+ → \s+
        val cleaned = r.replace("""\s+(?!\S)|\s+""", """\s+""")
        java.util.regex.Pattern.compile(cleaned)
      }

    Tokenizer(vocab, merges, mode, bosId, eosId, addBos, specialTokens, splitPattern)
