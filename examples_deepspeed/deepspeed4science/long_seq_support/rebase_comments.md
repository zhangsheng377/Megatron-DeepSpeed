# List of rebase efforts/achievements:
- Enabling Megatron-LM's sequence parallel
- Enabling rotary positional embedding
- Enabling FlashAttention v1 and v2
- Fix the conflicts related to activation checkpointing when DeepSpeed is used with the newest Megatron-LM since NVIDIA introduced some new fine-grained partial checkpointing techniques. DeepSpeed is not compatible with that.
- Major refactor to DeepSpeed pipeline parallelism implementation for GPT model in order to work with newest Megatron-LM
- Fix model checkpoint save/load when DeepSpeed is used with the newest Megatron-LM
- Fully verified the performance and correctness of GPT pretraining after rebasing
