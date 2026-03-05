import layoutz.*

// ── Model ──────────────────────────────────────────────────────────────────

case class ChatMessage(role: String, content: String)

case class ChatState(
    messages:         Vector[ChatMessage] = Vector.empty,
    inputText:        String              = "",
    isGenerating:     Boolean             = false,
    partialResponse:  String              = "",
    tokensGenerated:  Int                 = 0,
    startTimeMs:      Long                = 0L,
    error:            Option[String]      = None,
    totalTokens:      Int                 = 0,
    totalTimeMs:      Long                = 0L
)

// ── Messages ───────────────────────────────────────────────────────────────

sealed trait ChatMsg
case class KeyInput(c: Char) extends ChatMsg
case object Backspace extends ChatMsg
case object Submit extends ChatMsg
case object PollTokens extends ChatMsg
case class GenerationDone(result: Either[String, Unit]) extends ChatMsg

// ── Thread-safe token buffer ───────────────────────────────────────────────

object TokenBuffer:
  private val lock = new Object()
  private val buf  = scala.collection.mutable.ArrayBuffer[String]()

  def push(token: String): Unit = lock.synchronized { buf += token }

  def drain(): Vector[String] = lock.synchronized {
    val v = buf.toVector; buf.clear(); v
  }

  def reset(): Unit = lock.synchronized { buf.clear() }

// ── App ────────────────────────────────────────────────────────────────────

object ChatApp extends LayoutzApp[ChatState, ChatMsg]:

  // Set by main() before run()
  var model:     LlamaModel   = null
  var tokenizer: Tokenizer    = null
  var cfg:       LlamaConfig  = null
  var maxTokens: Int          = 512
  var temperature: Float      = 0.6f
  var topP: Float             = 1.0f
  var modelName: String       = "Qwen3-0.6B"

  private val SystemPrompt = "You are a helpful assistant. Keep responses concise."

  // ── Elm lifecycle ──────────────────────────────────────────────────────

  def init: (ChatState, Cmd[ChatMsg]) =
    (ChatState(), Cmd.none)

  def update(msg: ChatMsg, state: ChatState): (ChatState, Cmd[ChatMsg]) = msg match

    case KeyInput(c) =>
      if state.isGenerating then (state, Cmd.none)
      else (state.copy(inputText = state.inputText + c), Cmd.none)

    case Backspace =>
      if state.isGenerating || state.inputText.isEmpty then (state, Cmd.none)
      else (state.copy(inputText = state.inputText.dropRight(1)), Cmd.none)

    case Submit =>
      val text = state.inputText.trim
      if text.isEmpty || state.isGenerating then (state, Cmd.none)
      else if text == "/exit" || text == "exit" then (state, Cmd.exit)
      else if text == "/clear" then
        (state.copy(messages = Vector.empty, inputText = ""), Cmd.none)
      else
        val userMsg   = ChatMessage("user", text)
        val history   = state.messages :+ userMsg
        val prompt    = buildPrompt(history)
        TokenBuffer.reset()

        val cmd = Cmd.task {
          val promptTokens = tokenizer.encode(prompt, addBos = tokenizer.addBos)
          val rng = scala.util.Random()
          model.generate(
            promptTokens,
            maxNewTokens = maxTokens,
            eosId        = tokenizer.eosId,
            temperature  = temperature,
            topP         = topP,
            rng          = rng,
            onToken      = { id =>
              if id != tokenizer.eosId then
                val tok = tokenizer.decodeToken(id)
                TokenBuffer.push(tok)
            }
          )
        }(result => GenerationDone(result.map(_ => ())))

        val newState = state.copy(
          messages        = history,
          inputText       = "",
          isGenerating    = true,
          partialResponse = "",
          tokensGenerated = 0,
          startTimeMs     = System.currentTimeMillis(),
          error           = None
        )
        (newState, cmd)

    case PollTokens =>
      val tokens = TokenBuffer.drain()
      if tokens.isEmpty then (state, Cmd.none)
      else
        val appended = state.partialResponse + tokens.mkString
        (state.copy(
          partialResponse = appended,
          tokensGenerated = state.tokensGenerated + tokens.length
        ), Cmd.none)

    case GenerationDone(result) =>
      val response = state.partialResponse + TokenBuffer.drain().mkString
      val elapsed  = System.currentTimeMillis() - state.startTimeMs
      val assistantMsg = ChatMessage("assistant",
        if response.nonEmpty then response else "(empty response)")
      result match
        case Right(_) =>
          (state.copy(
            messages        = state.messages :+ assistantMsg,
            isGenerating    = false,
            partialResponse = "",
            tokensGenerated = 0,
            error           = None,
            totalTokens     = state.totalTokens + state.tokensGenerated,
            totalTimeMs     = state.totalTimeMs + elapsed
          ), Cmd.none)
        case Left(err) =>
          val finalMsg = if response.nonEmpty then
            ChatMessage("assistant", response + s"\n[error: $err]")
          else ChatMessage("assistant", s"[error: $err]")
          (state.copy(
            messages        = state.messages :+ finalMsg,
            isGenerating    = false,
            partialResponse = "",
            tokensGenerated = 0,
            error           = Some(err)
          ), Cmd.none)

  // ── Subscriptions ──────────────────────────────────────────────────────

  def subscriptions(state: ChatState): Sub[ChatMsg] =
    Sub.batch(
      Sub.onKeyPress {
        case Key.Enter     => Some(Submit)
        case Key.Backspace => Some(Backspace)
        case Key.Char(c)   => Some(KeyInput(c))
        case _             => None
      },
      if state.isGenerating then Sub.time.everyMs(50, PollTokens)
      else Sub.none
    )

  // ── View ───────────────────────────────────────────────────────────────

  def view(state: ChatState): Element =

    // ── Sidebar: model info ──────────────────────────────────────────────
    val dot = Text("●").color(Color.Green)
    val modelLabel = Text(s" $modelName").color(Color.Cyan).style(Style.Bold)

    val sidebar = box("Model")(
      layout(row(dot, modelLabel)),
      br,
      Text("Architecture").color(Color.Magenta).style(Style.Bold),
      kv(
        Text("Layers").color(Color.BrightBlack)    -> Text(s"${cfg.numHiddenLayers}"),
        Text("Hidden").color(Color.BrightBlack)     -> Text(s"${cfg.hiddenSize}"),
        Text("Q Heads").color(Color.BrightBlack)    -> Text(s"${cfg.numAttentionHeads}"),
        Text("KV Heads").color(Color.BrightBlack)   -> Text(s"${cfg.numKeyValueHeads}"),
        Text("Head dim").color(Color.BrightBlack)   -> Text(s"${cfg.headDim}"),
        Text("Vocab").color(Color.BrightBlack)      -> Text(s"${cfg.vocabSize}")
      ),
      br,
      Text("Sampling").color(Color.Magenta).style(Style.Bold),
      kv(
        Text("Temp").color(Color.BrightBlack)       -> Text(f"$temperature%.2f"),
        Text("Top-P").color(Color.BrightBlack)      -> Text(f"$topP%.2f"),
        Text("Max tok").color(Color.BrightBlack)    -> Text(s"$maxTokens")
      ),
      br,
      Text("Session").color(Color.Magenta).style(Style.Bold),
      kv(
        Text("Turns").color(Color.BrightBlack)      -> Text(s"${state.messages.count(_.role == "user")}"),
        Text("Tokens").color(Color.BrightBlack)     -> Text(s"${state.totalTokens}"),
        Text("Avg tok/s").color(Color.BrightBlack)  -> Text(
          if state.totalTimeMs > 100 then f"${state.totalTokens.toDouble / state.totalTimeMs * 1000}%.0f"
          else "—"
        )
      ),
      br,
      Text("/exit  quit").color(Color.BrightBlack),
      Text("/clear history").color(Color.BrightBlack)
    )

    // Compute right panel inner width to fill remaining terminal space.
    val sidebarW = sidebar.width
    // box adds 4 chars (border + padding on each side), columns adds 2 spacing
    val termW = try
      val raw = scala.sys.process.Process(Seq("sh", "-c", "stty size < /dev/tty")).!!.trim
      raw.split("\\s+")(1).toInt
    catch case _: Exception => 120
    val rightInnerW = math.max(30, termW - sidebarW - 2 - 4)

    // Word-wrap text to fit within a given width
    def wordWrap(text: String, width: Int): Seq[String] =
      if text.length <= width then Seq(text)
      else
        val lines = scala.collection.mutable.ArrayBuffer.empty[String]
        val words = text.split(" ")
        val line = new StringBuilder
        for w <- words do
          if line.isEmpty then line.append(w)
          else if line.length + 1 + w.length <= width then
            line.append(" "); line.append(w)
          else
            lines += line.toString; line.clear(); line.append(w)
        if line.nonEmpty then lines += line.toString
        lines.toSeq

    val labelW = 7 // " You " or "  AI " + " " = 6-7 chars
    val textW = rightInnerW - labelW

    // ── Chat area ────────────────────────────────────────────────────────
    val chatLines = scala.collection.mutable.ArrayBuffer.empty[Element]

    if state.messages.isEmpty && !state.isGenerating then
      chatLines += Text("Start a conversation...").color(Color.BrightBlack)
      chatLines += Text("")

    state.messages.foreach { msg =>
      msg.role match
        case "user" =>
          val wrapped = wordWrap(msg.content, textW)
          chatLines += layout(row(
            Text(" You ").bg(Color.Green).color(Color.Black).style(Style.Bold),
            Text(" "),
            Text(wrapped.head)
          ))
          wrapped.tail.foreach { line =>
            chatLines += Text(s"${" " * labelW}$line")
          }
        case "assistant" =>
          val wrapped = wordWrap(msg.content, textW)
          chatLines += layout(row(
            Text("  AI ").bg(Color.Blue).color(Color.Black).style(Style.Bold),
            Text(" "),
            Text(wrapped.head).color(Color.White)
          ))
          wrapped.tail.foreach { line =>
            chatLines += Text(s"${" " * labelW}$line").color(Color.White)
          }
        case _ =>
          chatLines += Text(s"[${msg.role}] ${msg.content}")
      chatLines += Text("")
    }

    // Streaming response
    if state.isGenerating then
      val elapsed = (System.currentTimeMillis() - state.startTimeMs).toDouble / 1000.0
      val tokS = if elapsed > 0.1 then f"${state.tokensGenerated / elapsed}%.0f"
                 else "—"
      val cursor = if (System.currentTimeMillis() / 500) % 2 == 0 then "█" else " "
      val wrapped = wordWrap(state.partialResponse, textW)
      chatLines += layout(row(
        Text("  AI ").bg(Color.Blue).color(Color.Black).style(Style.Bold),
        Text(" "),
        Text((if wrapped.nonEmpty then wrapped.head else "") + cursor).color(Color.White)
      ))
      if wrapped.length > 1 then
        wrapped.tail.foreach { line =>
          chatLines += Text(s"${" " * labelW}$line").color(Color.White)
        }
      chatLines += Text(s"   $tokS tok/s | ${state.tokensGenerated} tokens").color(Color.BrightBlack)

    // A ruler line of spaces forces the box to be at least rightInnerW wide
    val ruler = Text(" " * rightInnerW)

    val chatArea = box("Chat")(Layout(chatLines.toSeq :+ ruler))

    // ── Input bar ────────────────────────────────────────────────────────
    val inputContent =
      if state.isGenerating then
        val spinFrames = Array("⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏")
        val frame = spinFrames(((System.currentTimeMillis() / 80) % spinFrames.length).toInt)
        layout(row(
          Text(s" $frame ").color(Color.Yellow),
          Text("generating...").color(Color.BrightBlack)
        ))
      else
        layout(row(
          Text(" > ").color(Color.Green).style(Style.Bold),
          Text(state.inputText + "█")
        ))

    val inputBar = box("")(Layout(Seq(inputContent, ruler)))

    // ── Status bar ───────────────────────────────────────────────────────
    val statusBar = layout(row(
      Text(" Ctrl+Q ").color(Color.BrightBlack),
      Text("quit").color(Color.BrightBlack),
      Text("  Enter ").color(Color.BrightBlack),
      Text("send").color(Color.BrightBlack),
      Text("  /exit ").color(Color.BrightBlack),
      Text("quit").color(Color.BrightBlack),
      Text("  /clear ").color(Color.BrightBlack),
      Text("reset").color(Color.BrightBlack)
    ))

    // ── Compose layout: [sidebar | chat+input] ──────────────────────────
    val rightPanel = layout(chatArea, inputBar)

    layout(
      Text(" scala-mlx ").bg(Color.Cyan).color(Color.Black).style(Style.Bold),
      br,
      columns(sidebar, rightPanel),
      statusBar
    )

  // ── Helpers ────────────────────────────────────────────────────────────

  private def buildPrompt(messages: Vector[ChatMessage]): String =
    val sb = new StringBuilder
    sb.append(s"<|im_start|>system\n$SystemPrompt<|im_end|>\n")
    messages.foreach { msg =>
      sb.append(s"<|im_start|>${msg.role}\n${msg.content}<|im_end|>\n")
    }
    // Skip chain-of-thought
    sb.append("<|im_start|>assistant\n<think>\n\n</think>\n")
    sb.toString

// ── Supported models ────────────────────────────────────────────────────

case class ModelInfo(
    id:          String,   // HuggingFace repo id
    name:        String,   // Short display name
    params:      String,   // Parameter count
    size:        String,   // Approximate download size
    description: String    // Short description
)

object SupportedModels:
  val models: Seq[ModelInfo] = Seq(
    ModelInfo("mlx-community/Qwen3-0.6B-4bit",       "Qwen3-0.6B",   "0.6B", "~335 MB",  "Fast, lightweight chat model (default)"),
    ModelInfo("mlx-community/Qwen3-1.7B-4bit",       "Qwen3-1.7B",   "1.7B", "~1.0 GB",  "Better quality, still fast on M-series"),
    ModelInfo("mlx-community/Qwen3-4B-4bit",         "Qwen3-4B",     "4B",   "~2.3 GB",  "Strong reasoning, good for complex tasks"),
    ModelInfo("mlx-community/SmolLM2-135M-Instruct", "SmolLM2-135M", "135M", "~270 MB",  "Tiny model, very fast, limited quality"),
    ModelInfo("mlx-community/Llama-3.2-1B-Instruct-4bit", "Llama-3.2-1B", "1B", "~700 MB", "Meta Llama 3.2, good general purpose"),
  )

  def find(query: String): Option[ModelInfo] =
    models.find(m =>
      m.id.equalsIgnoreCase(query) ||
      m.name.equalsIgnoreCase(query) ||
      m.id.split("/").last.equalsIgnoreCase(query)
    )

  def printList(): Unit =
    println("Supported models:\n")
    println(f"  ${"Name"}%-20s ${"Params"}%-8s ${"Size"}%-10s ${"Description"}")
    println("  " + "─" * 80)
    models.foreach { m =>
      println(f"  ${m.name}%-20s ${m.params}%-8s ${m.size}%-10s ${m.description}")
    }

  def printUsage(): Unit =
    println("Usage: ./run-demo.sh <command> [options]\n")
    println("Commands:")
    println("  --list-models                List available models")
    println("  --download <model>           Download a model")
    println("  --model <model>              Start chat with a specific model")
    println("")
    println("Options:")
    println("  --max-tokens <n>             Max tokens to generate (default: 512)")
    println("  --temperature <f>            Sampling temperature (default: 0.6)")
    println("  --top-p <f>                  Top-p sampling (default: 1.0, use < 1.0 for nucleus)")
    println("")
    println("Examples:")
    println("  ./run-demo.sh --list-models")
    println("  ./run-demo.sh --download Qwen3-0.6B")
    println("  ./run-demo.sh --model Qwen3-0.6B")
    println("  ./run-demo.sh --model Qwen3-0.6B --max-tokens 1024")

// ── Entry point ──────────────────────────────────────────────────────────

object ChatDemo:

  def main(args: Array[String]): Unit =
    val projectDir = os.Path(sys.env.getOrElse("PROJECT_DIR", os.pwd.toString))
    val modelsDir  = projectDir / "models"

    // Show usage if no arguments
    if args.isEmpty then
      SupportedModels.printUsage()
      sys.exit(0)

    // Parse flags
    var maxTokens   = 512
    var temperature = 0.6f
    var topP        = 1.0f
    var modelName: Option[String] = None
    var listModels  = false
    var downloadModel: Option[String] = None
    var i = 0
    while i < args.length do
      args(i) match
        case "--max-tokens"  if i + 1 < args.length => maxTokens     = args(i+1).toInt;   i += 2
        case "--temperature" if i + 1 < args.length => temperature   = args(i+1).toFloat; i += 2
        case "--top-p"       if i + 1 < args.length => topP          = args(i+1).toFloat; i += 2
        case "--model"       if i + 1 < args.length => modelName     = Some(args(i+1));   i += 2
        case "--download"    if i + 1 < args.length => downloadModel = Some(args(i+1));   i += 2
        case "--list-models" => listModels = true; i += 1
        case other =>
          System.err.println(s"Unknown option: $other")
          System.err.println("")
          SupportedModels.printUsage()
          sys.exit(1)

    // ── List models ─────────────────────────────────────────────────────
    if listModels then
      SupportedModels.printList()
      sys.exit(0)

    // ── Download model ──────────────────────────────────────────────────
    downloadModel.foreach { query =>
      SupportedModels.find(query) match
        case None =>
          System.err.println(s"Unknown model: $query")
          System.err.println("Use --list-models to see available models.")
          sys.exit(1)
        case Some(info) =>
          val destDir = modelsDir / info.id.split("/").last
          if os.exists(destDir / "model.safetensors") then
            println(s"Model ${info.name} already downloaded at $destDir")
          else
            println(s"Downloading ${info.name} (${info.size}) from ${info.id}...")
            os.makeDir.all(destDir)
            val rc = os.proc(
              "huggingface-cli", "download", info.id,
              "config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors",
              "--local-dir", destDir.toString,
              "--local-dir-use-symlinks", "False"
            ).call(
              stdout = os.Inherit,
              stderr = os.Inherit,
              check = false
            )
            if rc.exitCode != 0 then
              System.err.println("Download failed.")
              sys.exit(1)
            println(s"Downloaded ${info.name} to $destDir")
          sys.exit(0)
    }

    // ── Require --model ─────────────────────────────────────────────────
    if modelName.isEmpty then
      System.err.println("Error: --model <model> is required to start chat.")
      System.err.println("")
      SupportedModels.printUsage()
      sys.exit(1)

    // ── Resolve model directory ─────────────────────────────────────────
    val modelDir: os.Path = modelName match
      case Some(query) =>
        SupportedModels.find(query) match
          case Some(info) =>
            val dir = modelsDir / info.id.split("/").last
            if !os.exists(dir / "model.safetensors") then
              System.err.println(s"Model ${info.name} not downloaded. Run: ./run-demo.sh --download ${info.name}")
              sys.exit(1)
            dir
          case None =>
            // Try as a direct path
            val dir = os.Path(query, os.pwd)
            if !os.exists(dir / "model.safetensors") then
              System.err.println(s"Unknown model: $query")
              System.err.println("Use --list-models to see available models.")
              sys.exit(1)
            dir
      case None =>
        sys.exit(1) // unreachable, handled above

    val configFile    = modelDir / "config.json"
    val tokenizerFile = modelDir / "tokenizer.json"
    val weightsFile   = modelDir / "model.safetensors"

    Seq(configFile, tokenizerFile, weightsFile).foreach { f =>
      if !os.exists(f) then
        System.err.println(s"Missing: $f")
        System.err.println("Run --download <model> to download a model.")
        sys.exit(1)
    }

    System.err.println(s"Loading model from $modelDir ...")
    val cfg       = LlamaConfig.fromFile(configFile)
    val tokenizer = Tokenizer.fromFile(tokenizerFile, cfg)
    val model     = LlamaModel(cfg, weightsFile)
    System.err.println("Model loaded. Starting chat...")

    // Derive display name from directory or model info
    val displayName = SupportedModels.models
      .find(m => modelDir.last == m.id.split("/").last)
      .map(_.name)
      .getOrElse {
        val paramEst = cfg.numHiddenLayers * cfg.hiddenSize * cfg.hiddenSize * 12L / 1_000_000_000.0
        f"${modelDir.last} (${paramEst}%.1fB)"
      }

    try
      ChatApp.model       = model
      ChatApp.tokenizer   = tokenizer
      ChatApp.cfg         = cfg
      ChatApp.maxTokens   = maxTokens
      ChatApp.temperature = temperature
      ChatApp.topP        = topP
      ChatApp.modelName   = displayName

      ChatApp.run(
        clearOnStart     = true,
        clearOnExit      = true,
        showQuitMessage  = false,
        renderIntervalMs = 50
      )
    finally
      model.close()
