# AIstotle — The Modern Philosopher

A transformer LLM built from scratch in PyTorch and trained on classical philosophy texts — Aristotle, Plato, Marcus Aurelius, Nietzsche, Seneca, and others.

## What I Built

- **BPE Tokenizer** — built from scratch with a doubly linked list to speed up vocabulary training
- **Transformer Architecture** — multi-head attention, feed-forward layers, residual connections, and layer normalization
- **Data Pipeline** — collected and cleaned 13 philosophy texts from Project Gutenberg
- **Training Loop** — cross-entropy loss, Adam optimizer, batch sampling
- **Text Generation** — generates text token by token using temperature sampling

## Challenges & Solutions

- **Tokenization approach** — debated between word, character, and subword tokenization. Built BPE from scratch because subword gives the model better semantic understanding than characters, and it was a good learning experience
- **BPE training speed** — rescanning the full token list to count pairs each iteration was too slow. Built a doubly linked list to track neighbors, only updating affected pairs after each merge resulting in a **1700x speedup**
- **Overfitting** — model initially memorized training text and generated gibberish. Added dropout layers and reduced batch size to encourage learning general patterns over memorization

## Stack
- Python, PyTorch
- Trained on Apple M3 Pro

## Roadmap
- [x] Data pipeline
- [x] BPE tokenizer
- [x] Transformer architecture
- [x] Training loop
- [x] Inference
- [ ] Instruction fine tuning for coherent Q&A
- [ ] Larger model