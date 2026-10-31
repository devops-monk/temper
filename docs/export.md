# Export

When a run finishes, open **Export**, pick the run (or one of its checkpoints) and the formats you want:

| Format | What it is | Use it with |
|--------|-----------|-------------|
| GGUF | the merged model, quantized | Temper, llama.cpp, Ollama, LM Studio |
| MLX | the merged model for Apple Silicon | Temper, `mlx-lm` |
| LoRA adapter | just the trained adapter, a few MB | sharing, or loading on top of the base model |
| Merged 16-bit | full-precision safetensors | further training, other converters, vLLM |
| Merged 4-bit | 4-bit safetensors | smaller downloads for PyTorch users |

## GGUF quants

Q4_K_M is the usual choice: about a quarter of the 16-bit size with little loss. Q5_K_M, Q6_K and Q8_0 keep more
quality; Q3_K_M and Q2_K trade quality for size. You can pick several in one export.

*Calibrate on the training data (importance matrix)* measures which weights matter most on your own
examples. It helps the small quants most, and adds the IQ4_XS, IQ3_M and IQ2_M quants, which need it.

## After exporting

- **Chat** loads a GGUF export straight into Chat to try it.
- **Add to Ollama** copies a GGUF into Ollama under a name you choose, so `ollama run` and apps that use
  Ollama can run it.
- **Publish to Hugging Face**, on the Library page, uploads a run's export or adapter with a model card written from the run
  (base model, data and settings). Sign in first under Settings → Hugging Face account. Prepared datasets
  can be published the same way.

Next: [connect your tools](connect.md).
