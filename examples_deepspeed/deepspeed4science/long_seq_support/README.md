# Megatron Sequence Parallelism

We rebased and enabled DeepSpeed with the newest Megatron for long sequence support. This folder contains examples that demonstrate how to use new Megatron-DeepSpeed's sequence parallelism.

## Rebasing Efforts/Achievements
- Enabled Megatron-LM's sequence parallel
- Enabled rotary positional embedding
- Enabled FlashAttention v1 and v2
- Fixed the conflicts related to activation checkpointing when DeepSpeed was used with the newest Megatron-LM since NVIDIA introduced some new fine-grained partial checkpointing techniques. DeepSpeed was not compatible with that
- Major refactor to DeepSpeed pipeline parallelism implementation for GPT model in order to work with newest Megatron-LM
- Fixed model checkpoint save/load when DeepSpeed was used with the newest Megatron-LM
- First generated attention mask on CPU memory and then moved it into GPU memory to avoid out of memory error when large sequence length
- Split weights of position encoding across all GPUs when enabling sequence parallel
- Fully verified the performance and correctness of GPT pretraining after rebasing

## Setting Up the Virtual Environment

```shell
# clone source code
git clone https://github.com/microsoft/DeepSpeed.git
git clone https://github.com/microsoft/Megatron-DeepSpeed.git
git clone https://github.com/NVIDIA/apex

# creat a new virtual environment
cd Megatron-DeepSpeed
python3 -m venv ./venvs/megatron-deepspeed --system-site-packages
source ./venvs/megatron-deepspeed/bin/activate

# install the newest DeepSpeed
cd ../DeepSpeed/
pip install -e .

# install apex
cd ../apex/
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" -e ./

# install pybind11
cd ../
pip install pybind11

# install Triton-based FlashAttention
git clone -b legacy-backend https://github.com/openai/triton
cd triton/python/
pip install cmake
pip install .

cd ../
git clone -b v1.0.4 https://github.com/HazyResearch/flash-attention
cd flash-attention
python setup.py install
```

## Enabling Sequence Parallelism

To enable sequence parallelism, set `--sequence-parallel` argument. The the degree of sequence parallelism is equal to the degree of model tensor parallelism. Ensure that the sequence length is divisible by the degree of sequence parallelism. 
Ensure your model configuration is compliant with FlashAttention's requirements. For instance, to achieve optimal performance, the head size should be divisible by 8. Refer to the document of [FlashAttention](https://github.com/Dao-AILab/flash-attention/tree/v1.0.4) for more details.
Some working examples ([GPT1.3B](pretrain_gpt_1.3B_seq_parallel.sh) [GPT30B](pretrain_gpt_13B_seq_parallel.sh)) that enable sequence parallelism, are available in this foloder.

## Max Sequence Length and Throughput Comparison between Old Megatron-DeepSpeed and New Megatron-DeepSpeed

Experiments are performed on 4 NVIDIA DGX A100-40GB nodes, connected through 8 HDR InfiniBand (200Gb/s per HDR). Due to the lack of FlashAttention and memory optimization. The max sequence length and throughput of old Megatron-DeepSpeed are quite limited. TP is short for tensor parallel. 

| Sequence Length | Old Megatron-DeepSpeed  (TFLOPS) | New Megatron-DeepSpeed  (TFLOPS) |
|-----------------|----------------------------------|----------------------------------|
| 2k              | 25 (TP=32)                       | 68 (TP size=32)                  |
| 4k              | 28 (TP=32)                       | 80 (TP size=32)                  |
| 8k              | OoM                              | 86 (TP size=32)                  |
| 16k             | OoM                              | 92 (TP size=32)                  |
| 32k             | OoM                              | 100 (TP size=32)                 |
| 64k             | OoM                              | 106 (TP size=32)                 |
| 128k            | OoM                              | 119 (TP size=32)                 |
| 256k            | OoM                              | 94 (TP size=32)                  |

## Acknowledgements

This research used resources of the Argonne Leadership Computing Facility, which is a DOE Office of Science User Facility supported under Contract DE-AC02-06CH11357.
Especially, used Argonne Leadership Computing Facility (ALCF) [ThetaGPU](https://www.alcf.anl.gov/support-center/training-assets/getting-started-theta-and-thetagpu) supercomputer. 