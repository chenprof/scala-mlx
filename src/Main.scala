import scala.scalanative.unsafe.*

object Main:

  private val ModelDir      = os.pwd / "model"
  private val ConfigFile    = ModelDir / "config.json"
  private val TokenizerFile = ModelDir / "tokenizer.json"
  private val WeightsFile   = ModelDir / "model.safetensors"

  def main(args: Array[String]): Unit =

    // Parse flags: [--temperature T] [--top-p P] [--seed N] [--max-tokens N] prompt...
    var temperature = 0.0f
    var topP        = 1.0f
    var seed        = -1L          // -1 = random seed
    var maxTokens   = 512
    val promptParts = scala.collection.mutable.ArrayBuffer.empty[String]
    var i = 0
    while i < args.length do
      args(i) match
        case "--temperature" if i + 1 < args.length => temperature = args(i+1).toFloat; i += 2
        case "--top-p"       if i + 1 < args.length => topP        = args(i+1).toFloat; i += 2
        case "--seed"        if i + 1 < args.length => seed        = args(i+1).toLong;  i += 2
        case "--max-tokens"  if i + 1 < args.length => maxTokens   = args(i+1).toInt;  i += 2
        case "--" => promptParts ++= args.drop(i + 1); i = args.length
        case other => promptParts += other; i += 1

    val userPrompt = if promptParts.nonEmpty then promptParts.mkString(" ")
                     else "Tell me something interesting about Apple Silicon."

    Seq(ConfigFile, TokenizerFile, WeightsFile).foreach { f =>
      if !os.exists(f) then
        System.err.println(s"Missing: $f")
        System.err.println("Run ./setup.sh first to build libraries and download the model.")
        sys.exit(1)
    }

    val cfg       = LlamaConfig.fromFile(ConfigFile)
    val tokenizer = Tokenizer.fromFile(TokenizerFile, cfg)

    // Qwen3 chat template. Pre-fill <think>\n\n</think> to skip chain-of-thought mode
    // and get a direct answer. These are special tokens (ids 151667/151668) so the
    // model sees the completed think block and proceeds straight to the response.
    val prompt =
      s"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" +
      s"<|im_start|>user\n$userPrompt<|im_end|>\n" +
      s"<|im_start|>assistant\n<think>\n\n</think>\n"

    val rng = if seed >= 0 then scala.util.Random(seed) else scala.util.Random()

    val samplingDesc =
      if temperature == 0.0f then "greedy"
      else if topP < 1.0f    then f"temp=${temperature}%.2f  top-p=${topP}%.2f"
      else                        f"temp=${temperature}%.2f"

    println(s"Model   : ${cfg.numHiddenLayers} layers  hidden=${cfg.hiddenSize}  " +
            s"heads=${cfg.numAttentionHeads}  head_dim=${cfg.headDim}  vocab=${cfg.vocabSize}")
    println(s"Sampling: $samplingDesc")
    println(s"Prompt  : $userPrompt")
    println()

    val promptTokens = tokenizer.encode(prompt, addBos = tokenizer.addBos)

    val t0 = System.currentTimeMillis()
    var tokenCount = 0

    val model = LlamaModel(cfg, WeightsFile)
    try
      model.generate(
        promptTokens,
        maxNewTokens = maxTokens,
        eosId        = tokenizer.eosId,
        temperature  = temperature,
        topP         = topP,
        rng          = rng,
        onToken      = { id =>
          val tok = tokenizer.decodeToken(id)
          print(tok)
          Console.flush()
          tokenCount += 1
        }
      )
    finally
      model.close()

    val elapsed = (System.currentTimeMillis() - t0) / 1000.0
    println()
    println()
    println(f"Generated $tokenCount tokens in ${elapsed}%.1f s  (${tokenCount / elapsed}%.1f tok/s)")
