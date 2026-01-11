
## About this folder

This folder contains a small, self-contained example of applying LoRA (Low-Rank Adaptation) to a GPT-style model (distilgpt2) using the Hugging Face Transformers + PEFT libraries. It is intended as an educational, experimental demo to show how to:

- Attach LoRA adapters to a frozen base model
- Train only adapter parameters on a small dataset
- Save and reload the adapter-only checkpoint for inference

Files of interest

- `train.py` — Full training script that: loads the dataset, tokenizes, attaches LoRA adapters (PEFT), trains with the HF Trainer, and saves the adapter files into `lora-quote-model/`.
- `test_model.py` — Minimal example that shows how to load the base model and then overlay the LoRA adapter with `PeftModel.from_pretrained()` for inference.
- `README.MD` — (this file) explains the contents and usage notes.

See the `Output` section above for details about the files produced by training (adapter weights, adapter config, tokenizer files, and training snapshot).

## Install requirements

1. Install PyTorch with CUDA 12.8 support (adjust if needed):
```bash
uv sync
```

2. Verify PyTorch installation:
```uv run python - <<EOF
import torch
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("Arch:", torch.cuda.get_arch_list())
print("GPU:", torch.cuda.get_device_name(0))
EOF
```
Example of output:
```text
Torch: 2.11.0.dev20260111+cu128
CUDA: 12.8
Arch: [..., 'sm_120']
GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
```

3. If verification is not successful then install this:
```bash
export UV_HTTP_TIMEOUT=300
uv pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128
```

And verify again

## Usage tips

- This example is small on purpose. Use it to learn how LoRA/PEFT integrate with HF models, not as a production training pipeline.
- If you have limited VRAM, reduce `per_device_train_batch_size` and/or use gradient accumulation.
- The script already uses `device_map="auto"` and bf16 where available to reduce memory usage; adjust only if you understand the tradeoffs.

## Limitations & disclaimer

This code and the provided training example are experimental and provided for educational purposes only. They are NOT intended for production use. By using these scripts you acknowledge:

- The example is simplified and omits many production concerns (robust logging, checkpoint management, data validation, secure model distribution, etc.).
- There is no warranty. Results are illustrative and may not generalize.
- Respect the licenses and terms of any datasets or base models you use.

If you plan to use these techniques in production, please consult the official PEFT and Hugging Face documentation and follow best practices for model evaluation, safety, and deployment.

## Output model / Result

Result is stored in `output_dir`

What exactly is in output_dir?

The structure of the output dir is the following:
```text
lora-quote-model/
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── training_args.bin
```

### adapter_model.safetensors

This contains:

* -ONLY the LoRA weights
* ~400k parameters
* A few MB

It does NOT contain GPT-2 weights.

Think of it as:
“A patch that modifies GPT-2’s behavior.”

### adapter_config.json

Contains info for PEFT (Parameter-Efficient Fine-Tuning):

* Which layers were modified
* LoRA rank (r=8)
* Scaling (α=16)
* Fan-in/fan-out settings

This is how PEFT knows how to re-attach LoRA later.


### Tokenizer files

Copied for convenience:

* Ensures consistent tokenization
* Important if you share the model

### training_args.bin

A snapshot of:
* Learning rate
* Batch size
* Epochs
* Precision mode

Useful for:
* Reproducibility
* Debugging

## How inference works after training

When you load the model later like this:
```python
base = AutoModelForCausalLM.from_pretrained("distilgpt2")
model = PeftModel.from_pretrained(base, "./lora-quote-model")
```

then internally this is happening
```text
Effective model = base_model + LoRA adapters
```

So:
* No copying.
* No merging.
* No rewriting weights.

* It’s dynamic composition.


## Contributing

Small improvements and corrections are welcome. Open a PR with a focused change (docs, bugfixes, or small script improvements). For large changes (e.g., adding full CI/test harness, or converting to a production training pipeline) please open an issue first to discuss scope.

## References

- PEFT (Parameter-Efficient Fine-Tuning) — https://github.com/huggingface/peft
- Hugging Face Transformers — https://huggingface.co/docs/transformers



