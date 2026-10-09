![GPT-2 Inference](public/gpt.png)

# GPT-2 Inference in C++

A project to understand how GPT-2 works internally by implementing its inference process in C++.

The goal is to build the model step by step and understand the operations involved in processing tokens and generating text, rather than relying entirely on a deep learning framework.

## Current Progress

- Loaded GPT-2 token embedding weights
- Loaded positional embedding weights
- Combined token and positional embeddings
- Loaded LayerNorm weights and biases
- Implemented LayerNorm in C++
- Tested the implementation with multiple tokens

## Tech Stack

- C++
- Python
- GPT-2 pretrained weights

## Project Structure

```text
gpt2/
├── public/
│   └── gpt.png
├── inference/
│   └── main.cc
├── modelloader/
└── weights/
```

## Build and Run

Compile the C++ implementation:

```bash
g++ -std=c++20 main.cc -o main.exe
```

Run on Windows:

```powershell
.\main.exe
```

Run these commands from the `inference` directory.

## Next Steps

- Implement Query, Key, and Value projections
- Implement causal self-attention
- Implement multi-head attention
- Implement the feed-forward network
- Complete the transformer blocks
- Implement next-token prediction
- Support user input and text generation

This project is a work in progress. The focus is on understanding the mathematics and implementation details behind transformer-based language models.