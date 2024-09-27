#!/bin/sh

if ! test -f llamafile; then
	curl -o llamafile 'https://huggingface.co/Mozilla/Mistral-7B-Instruct-v0.2-llamafile/resolve/main/mistral-7b-instruct-v0.2.Q4_0.llamafile?download=true'
	chmod +x llamafile
fi
