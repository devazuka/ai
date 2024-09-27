#!/bin/sh

./llamafile --server --api-key $PASSWORD -c 4096 --port $PORT

# llamafile server options:
# 
# --threads N, -t N:
# 	Set the number of threads to use during generation.
# 
# -tb N, --threads-batch N:
# 	Set the number of threads to use during batch and prompt processing.
#   If not specified, the number of threads will be set to the number of threads used for generation.
# 
# -m FNAME, --model FNAME:
# 	Specify the path to the LLaMA model file (e.g., models/7B/ggml-model.gguf).
# 
# -a ALIAS, --alias ALIAS:
# 	Set an alias for the model. The alias will be returned in API responses.
# 
# -c N, --ctx-size N:
# 	Set the size of the prompt context. The default is 512, but LLaMA models were built with a context of 2048,
#   which will provide better results for longer input/inference. The size may differ in other models, for example,
#   baichuan models were build with a context of 4096.
# 
# -ngl N, --n-gpu-layers N:
# 	When compiled with appropriate support (currently CLBlast or cuBLAS),
#   this option allows offloading some layers to the GPU for computation.
#   Generally results in increased performance.
# 
# -mg i, --main-gpu i:
# 	When using multiple GPUs this option controls which GPU is used for small tensors for which the overhead
#   of splitting the computation across all GPUs is not worthwhile.
#   The GPU in question will use slightly more VRAM to store a scratch buffer for temporary results.
#   By default GPU 0 is used. Requires cuBLAS.
# 
# -ts SPLIT, --tensor-split SPLIT:
# 	When using multiple GPUs this option controls how large tensors should be split across all GPUs.
#   SPLIT is a comma-separated list of non-negative values that assigns the proportion of data that each GPU should get in order.
#   For example, "3,2" will assign 60% of the data to GPU 0 and 40% to GPU 1.
#   By default the data is split in proportion to VRAM but this may not be optimal for performance. Requires cuBLAS.
# 
# -b N, --batch-size N:
# 	Set the batch size for prompt processing. Default: 512.
# 
# --memory-f32:
# 	Use 32-bit floats instead of 16-bit floats for memory key+value. Not recommended.
# 
# --mlock:
# 	Lock the model in memory, preventing it from being swapped out when memory-mapped.
# 
# --no-mmap:
# 	Do not memory-map the model. By default, models are mapped into memory,
#   which allows the system to load only the necessary parts of the model as needed.
# 
# --numa:
# 	Attempt optimizations that help on some NUMA systems.
# 
# --lora FNAME:
# 	Apply a LoRA (Low-Rank Adaptation) adapter to the model (implies --no-mmap).
#   This allows you to adapt the pretrained model to specific tasks or domains.
# 
# --lora-base FNAME:
# 	Optional model to use as a base for the layers modified by the LoRA adapter.
#   This flag is used in conjunction with the --lora flag, and specifies the base model for the adaptation.
# 
# -to N, --timeout N:
# 	Server read/write timeout in seconds. Default 600.
# 
# --host:
# 	Set the hostname or ip address to listen. Default 127.0.0.1.
# 
# --port:
# 	Set the port to listen. Default: 8080.
# 
# --path:
# 	path from which to serve static files (default examples/server/public)
# 
# --api-key:
# 	Set an api key for request authorization. By default the server responds to every request.
#   With an api key set, the requests must have the Authorization header set with the api key as Bearer token.
#   May be used multiple times to enable multiple valid keys.
# 
# --api-key-file:
# 	path to file containing api keys delimited by new lines.
#   If set, requests must include one of the keys for access. May be used in conjunction with --api-key's.
# 
# --embedding:
# 	Enable embedding extraction, Default: disabled.
# 
# -np N, --parallel N:
# 	Set the number of slots for process requests (default: 1)
# 
# -cb, --cont-batching:
# 	enable continuous batching (a.k.a dynamic batching) (default: disabled)
# 
# -spf FNAME, --system-prompt-file FNAME Set a file to load "a system prompt (initial prompt of all slots), this is useful for chat applications. See more
# 
# --mmproj MMPROJ_FILE:
# 	Path to a multimodal projector file for LLaVA.
# 
# --grp-attn-n:
# 	Set the group attention factor to extend context size through self-extend(default: 1=disabled), used together with group attention width --grp-attn-w
# 
# --grp-attn-w:
# 	Set the group attention width to extend context size through self-extend(default: 512), used together with group attention factor --grp-attn-n

