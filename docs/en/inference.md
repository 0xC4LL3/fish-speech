# Inference

The Fish Audio S2 model requires a large amount of VRAM. We recommend using a GPU with at least 24GB for inference. See [Running on 16GB GPUs](#running-on-16gb-gpus) if you have less.

## Download Weights

First, you need to download the model weights:

```bash
hf download fishaudio/s2-pro --local-dir checkpoints/s2-pro
```

## Command Line Inference

!!! note
    If you plan to let the model randomly choose a voice timbre, you can skip this step.

### 1. Get VQ tokens from reference audio

```bash
python fish_speech/models/dac/inference.py \
    -i "test.wav" \
    --checkpoint-path "checkpoints/s2-pro/codec.pth"
```

You should get a `fake.npy` and a `fake.wav`.

### 2. Generate Semantic tokens from text:

```bash
python fish_speech/models/text2semantic/inference.py \
    --text "The text you want to convert" \
    --prompt-text "Your reference text" \
    --prompt-tokens "fake.npy" \
    # --compile
```

This command will create a `codes_N` file in the working directory, where N is an integer starting from 0.

!!! note
    You may want to use `--compile` to fuse CUDA kernels for faster inference. However, we recommend using our sglang inference acceleration optimization.
    Correspondingly, if you do not plan to use acceleration, you can comment out the `--compile` parameter.

!!! info
    For GPUs that do not support bf16, you may need to use the `--half` parameter.

### 3. Generate vocals from semantic tokens:

```bash
python fish_speech/models/dac/inference.py \
    -i "codes_0.npy" \
```

After that, you will get a `fake.wav` file.

## Running on 16GB GPUs

By default S2 Pro allocates for its full 32768 token context at load time, whatever the
request actually needs. On a 16GB card that does not fit: the bf16 weights are 9.1GB, and
the context sized buffers add 5.9GB on top.

`--max-length` overrides the context window. The KV cache shrinks linearly with it and the
causal mask quadratically. Measured on an RTX 5080 (16GB) with about 0.8GB already taken by
the desktop. Peak VRAM here is the text2semantic model alone, so add roughly 1.7GB for the
codec if you are running the WebUI or the API server:

| max_length | Causal mask | KV cache | Peak VRAM | Tokens/sec |
| --- | --- | --- | --- | --- |
| 32768 (default) | 1.07 GB | 4.83 GB | out of memory | - |
| 24576 | 0.60 GB | 3.62 GB | 14.88 GB | 4.97 |
| 16384 | 0.27 GB | 2.42 GB | 12.82 GB | 6.51 |
| 8192 | 0.07 GB | 1.21 GB | 10.94 GB | 8.94 |

Throughput improves as the window shrinks because every decode step attends over the whole
window, not just the filled part. So `--max-length` is a speed setting as much as a memory
one.

The full WebUI at `--max-length 8192`, including the codec, peaks at 12.73GB.

Pick the window from how much audio one request needs. Audio codes run at about 21.5 tokens
per second, and the context holds the reference audio plus everything generated so far in
that request, so 8192 is worth roughly 4-5 minutes. Note that `generate_long` also rejects
any prompt longer than `max_length - 2048`.

```bash
# Command line
python fish_speech/models/text2semantic/inference.py --max-length 8192 ...

# WebUI
python tools/run_webui.py --max-length 8192 --compile

# API server
python tools/api_server.py --max-length 8192 --compile
```

Use `--compile` if you generate more than one clip per session. It costs about 145 seconds
of one time warmup and then runs roughly 4.5x faster: 39.9 tokens/sec instead of 8.9 at
`--max-length 8192`, which is about 1.8x faster than realtime. Servers pay the warmup once
at startup.

Keep bf16, which is the default. Do not pass `--half`; the 5080 has native bf16 and fp16
saves no memory here.

`--codec-precision bfloat16` takes the DAC codec from 1.7GB to about 0.9GB if you need a
little more headroom. It is not needed at `--max-length 8192`.

If you hit fragmentation related failures, start the process with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## WebUI Inference

### 1. Gradio WebUI

For compatibility, we still maintain the Gradio WebUI.

```bash
python tools/run_webui.py # --compile if you need acceleration
```

### 2. Awesome WebUI

Awesome WebUI is a modernized Web interface built with TypeScript, offering richer features and a better user experience.

**Build WebUI:**

You need to have Node.js and npm installed on your local machine or server.

1. Enter the `awesome_webui` directory:
   ```bash
   cd awesome_webui
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Build the WebUI:
   ```bash
   npm run build
   ```

**Start Backend Server:**

After building the WebUI, return to the project root and start the API server:

```bash
python tools/api_server.py --listen 0.0.0.0:8888 --compile
```

**Access:**

Once the server is running, you can access it via your browser:
`http://localhost:8888/ui`
