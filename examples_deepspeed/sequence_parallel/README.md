# Sequence Parallelism

The examples in this folder explain the usage of [DeepSpeed's sequence parallelism](http://LINK_TO_BLOG_POST_OR_PAPER/).

## Environment setup (FlashAttention)

Our sequence parallelism is designed to work with FlashAttention using Triton.
Refer to the following steps for installation.
Note that FlashAttention only supports Turing, Ampere, Ada, or Hopper GPUs.

```shell
WORK_DIR=flash_attn_repro
mkdir ${WORK_DIR} && cd ${WORK_DIR}
python -m venv venv/flash_attn_repro
source venv/flash_attn_repro/bin/activate
pip install packaging

# install triton
git clone -b legacy-backend https://github.com/openai/triton
cd triton/python/
pip install cmake; # build-time dependency
pip install .

# install
cd ${WORK_DIR}
git clone -b v1.0.4 https://github.com/HazyResearch/flash-attention
cd flash-attention
python setup.py install
```

## How to enable Sequence Parallelism

You can enable our sequence parallelim by setting the degree of parallelism to `--sequence-parallel-size`.
Note that tensor parallelism cannot be combined with this sequence parallelism.
The number of attention heads must also be divisible this number.

Please make sure your model configuration meets the requiments of FlashAttention. (For example, the head size must also be divisible by 8 for the best performance. See the document of [FlashAttention](https://github.com/Dao-AILab/flash-attention/tree/v1.0.4) for more details)

You can find working examples ([GPT1.3B](ds_pretrain_gpt_1.3B_seq_parallel.sh), [GPT30B](ds_pretrain_gpt_30B_seq_parallel.sh)) that enable sequence parallelism in this foloder.

# Benchmark results

## GPT 30B (32 A100s on 4 nodes)

The micro batch size was set to 1 and ZeRO stage 3 was enabled for all settings.
The sequence parallel sizes were chosen to achieve best TFLOPS.

| Sequence length | Global batch size | Sequence parallel size | TFLOPS |
| --------------- | ----------------- | -----------------------| ------ |
| 8k | 16 | 2 | 142 |
| 16k | 8 | 4 | 135 |
| 32k | 4 | 8 | 139 |
| 64k | 2 | 16 | 133 |
| 128k | 1 | 32 | 132 |
