# Megatron Sequence Parallelism

We rebased and enabled DeepSpeed with the newest Megatron for long sequence support. This folder contains examples that demonstrate how to use new Megatron-DeepSpeed's sequence parallelism.

## Rebasing Efforts/Achievements
- Enabling Megatron-LM's sequence parallel
- Enabling rotary positional embedding
- Enabling FlashAttention v1 and v2
- Fix the conflicts related to activation checkpointing when DeepSpeed is used with the newest Megatron-LM since NVIDIA introduced some new fine-grained partial checkpointing techniques. DeepSpeed is not compatible with that.
- Major refactor to DeepSpeed pipeline parallelism implementation for GPT model in order to work with newest Megatron-LM
- Fix model checkpoint save/load when DeepSpeed is used with the newest Megatron-LM
- First generate attention mask on CPU memory and then move it into GPU memory to avoid out of memory error when large sequence length
- Split weights of position encoding across all GPUs when enabling sequence parallel
- Fully verified the performance and correctness of GPT pretraining after rebasing

## Setting Up the Environment for FlashAttention

new Megatron-DeepSpeed's sequence parallelism can be combined with the following types of attention.

- Classic attention
- FlashAttention (enabled by `--use-flash-attn`)
- FlashAttention + Triton (enabled by `--use-flash-attn-triton`)

For the best performance, we recommend using FlashAttention + Triton. Here are the installation steps and the versions we have tested. Note that FlashAttention is compatible only with Turing, Ampere, Ada, or Hopper GPUs.

```shell
# install triton
git clone -b legacy-backend https://github.com/openai/triton
cd triton/python/
pip install cmake
pip install .

# install
cd ${WORK_DIR}
git clone -b v1.0.4 https://github.com/HazyResearch/flash-attention
cd flash-attention
python setup.py install
```

## Enabling Sequence Parallelism

To enable sequence parallelism, set `--sequence-parallel` argument. The the degree of sequence parallelism is equal to the degree of model tensor parallelism. Ensure that the sequence length is divisible by the degree of sequence parallelism. 
Ensure your model configuration is compliant with FlashAttention's requirements. For instance, to achieve optimal performance, the head size should be divisible by 8. Refer to the document of [FlashAttention](https://github.com/Dao-AILab/flash-attention/tree/v1.0.4) for more details.
Some working examples ([GPT1.3B](pretrain_gpt_1.3B_seq_parallel.sh) [GPT30B](pretrain_gpt_13B_seq_parallel.sh)) that enable sequence parallelism, are available in this foloder.

## Max Sequence Length and Throughput Comparison between Old Megatron-DeepSpeed and New Megatron-DeepSpeed

Due to the lack of FlashAttention and memory optimization. The max sequence length and throughput of old Megatron-DeepSpeed is quite limited. TP is short for tensor parallel. 

| Sequence Length | Old Megatron-DeepSpeed  (TFLOPS) | New Megatron-DeepSpeed  (TFLOPS) |
|-----------------|----------------------------------|----------------------------------|
| 2k              | 25 (TP=32)                       | 91 (TP size=16)                  |
| 4k              | 28 (TP=32)                       | 94 (TP size=16)                  |
| 8k              | OoM                              | 98 (TP size=16)                  |
| 16k             | OoM                              | 125 (TP size=16)                 |
| 32k             | OoM                              | 100 (TP size=16)                 |
| 64k             | OoM                              | 106 (TP size=32)                 |
| 128k            | OoM                              | 119 (TP size=32)                 |
| 256k            | OoM                              | 94 (TP size=32)                  |