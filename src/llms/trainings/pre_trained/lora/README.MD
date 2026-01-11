

## Output

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