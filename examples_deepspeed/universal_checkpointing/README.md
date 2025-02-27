# Universal Checkpoint examples

This folder contains example scripts that demonstrate how to use Universal Checkpoints to change the number of GPUs when training with ZeRO. With Universal Checkpoints, training can be resumed with a different parallelism degree on any of tensor slicing (TP), pipeline parallelism (PP), and data parallelism (DP). Using universal checkpoints involves the following three steps:

1. ZeRO-based training run, optionally combining TP and PP, that creates normal ZeRO checkpoints.  
2. Converting ZeRO checkpoint into the universal format using `ds_to_universal.py` utility of DeepSpeed.
3. Resuming training with the universal checkpoint, on a different number of GPUs.

## ZeRO stage 1 training
For ZeRO stage 1, we provide bash scripts for bf16 and fp16 training examples corresponding to the steps 1 and 3 above. The step 1 scripts launch a training run of TP=PP=DP=2 of 200 iterations that creates a checkpoint every 100 iterations. The step 3 scripts load a universal checkpoint of iteration 100 and resume training with TP=PP=2 and DP=1 for an additional 100 iterations. Users can modify these scripts to try out other save and resume 3D combinations (e.g., save TP=PP=DP=1 and resume TP=PP=DP=2). Tensorboard logs are created by both step 1 and 3 scripts to enable visual inspection of how well the loss curves of the initial and resumed training runs match, especially at iteration 101.  

1.  bf16:
    * run_bf16.sh: step 1
    * run_universal_bf16.sh: step 3

2. fp16:
    * run_fp16.sh: step 1 
    * run_universal_fp16.sh: step 3

Please note that these scripts should be run from the root folder of the repo (i.e., two levels above this README). For illustration, here are the commands for running the bf16 example. 

### Step 1: Create ZeRO checkpoint
```bash 
  bash examples_deepspeed/universal_checkpointing/run_bf16.sh 
```
By default the script will create the checkpoints in folder `z1_uni_ckpt/checkpoints/gpt2/z1/bf16/tp2_pp2_dp2_toy`

### Step 2: Convert ZeRO checkpoint of iteration 100 to Universal format
Assuming the DeepSpeed source code is cloned into the home folder, the following command will generate universal checkpoint for iteration 100. 

```bash
python ${HOME}/DeepSpeed/deepspeed/checkpoint/ds_to_universal.py \
    --input_folder z1_uni_ckpt/checkpoints/gpt2/z1/bf16/tp2_pp2_dp2_toy/global_step100 \
    --output_folder z1_uni_ckpt/checkpoints/gpt2/z1/bf16/tp2_pp2_dp2_toy/global_step100_universal
```
Note that we chose to create the universal checkpoint in the same checkpoint folder as the ZeRO checkpoint. This maintains the normal checkpoint folder structure expected by the Megatron-DeepSpeed code, which makes it easy to load universal checkpoints with little/no script or code changes. For clarity, we show below the contents of the checkpoint folder after creation of the universal checkpoint. Note that the conversion script creates `global_step100_universal` folder and `latest_universal` file.   

```bash
ls -l z1_uni_ckpt/checkpoints/gpt2/z1/bf16/tp2_pp2_dp2_toy/
total 48
drwxr-xr-x 2 user group  4096 Oct 21 08:51 global_step100
drwxr-xr-x 3 user group  4096 Oct 21 09:28 global_step100_universal
drwxr-xr-x 2 user group  4096 Oct 21 09:01 global_step200
-rw-r--r-- 1 user group    14 Oct 21 09:50 latest
-rw-r--r-- 1 user group     3 Oct 21 09:50 latest_checkpointed_iteration.txt
-rw-r--r-- 1 user group    24 Oct 21 09:28 latest_universal
-rwxr--r-- 1 user group 24177 Oct 21 09:50 zero_to_fp32.py
```

### Step3: Resume training with Universal checkpoint of iteration 100
```bash 
bash examples_deepspeed/universal_checkpointing/run_universal_bf16.sh
```
This resumption script effects the loading of universal checkpoint rather than the ZeRO checkpoint in the folder by passing `--universal-checkpoint` command line flag to the main training script (i.e., `pretrain_gpt.py`). 

## ZeRO stage 2 training (**Coming soon**)

## ZeRO stage 3 training (**Coming soon**)