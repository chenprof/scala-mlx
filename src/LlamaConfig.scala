import ujson.*

/** Parsed subset of config.json, compatible with any LLaMA-style model.
  *
  * head_dim: read explicitly if present (Qwen3 sets head_dim=128 independently
  * of hidden_size/num_heads), otherwise derived as hidden_size / num_heads.
  *
  * bos/eos token IDs: read from config so models like Qwen3 (bos=null,
  * eos=151645) are handled correctly without hardcoding LLaMA defaults.
  */
case class LlamaConfig(
    numHiddenLayers: Int,
    hiddenSize: Int,
    numAttentionHeads: Int,
    numKeyValueHeads: Int,
    headDim: Int,
    intermediateSize: Int,
    vocabSize: Int,
    rmsNormEps: Float,
    ropeTheta: Float,
    bosTokenId: Option[Int],
    eosTokenId: Int
):
  val ropeDims: Int = headDim

object LlamaConfig:

  def fromFile(path: os.Path): LlamaConfig =
    val json    = ujson.read(os.read(path))
    val numHeads = json("num_attention_heads").num.toInt
    val hidden   = json("hidden_size").num.toInt

    // Use explicit head_dim if present (Qwen3), else derive
    val headDim = json.obj.get("head_dim")
                    .map(_.num.toInt)
                    .getOrElse(hidden / numHeads)

    LlamaConfig(
      numHiddenLayers   = json("num_hidden_layers").num.toInt,
      hiddenSize        = hidden,
      numAttentionHeads = numHeads,
      numKeyValueHeads  = json("num_key_value_heads").num.toInt,
      headDim           = headDim,
      intermediateSize  = json("intermediate_size").num.toInt,
      vocabSize         = json("vocab_size").num.toInt,
      rmsNormEps        = json("rms_norm_eps").num.toFloat,
      ropeTheta         = json.obj.get("rope_theta").map(_.num.toFloat).getOrElse(10000.0f),
      bosTokenId        = json.obj.get("bos_token_id").flatMap(v => if v.isNull then None else Some(v.num.toInt)),
      eosTokenId        = json.obj.get("eos_token_id").map(_.num.toInt).getOrElse(2)
    )
