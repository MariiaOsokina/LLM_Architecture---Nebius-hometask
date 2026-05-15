# LLM_Architecture---Nebius-hometask

# Week 1 - Optimisation in PyTorch — Gradient Descent, SGD, Numerical Stability, and L1 Regularisation

Learning goals of this homework task:
* Understand preprocessing design choices (tokenisation provided, fixed vocabulary).
* Implement and train Logistic Regression manually in PyTorch using SGD.
* Explain why numerical stability matters in softmax and log-loss.
* Understand how optimisation parameters like learning rate and batch size affect training.
* Understand the effect of L1 regularisation and why it encourages sparsity.
* Understanding how an optimisation algorithm behaves when the loss function has different shapes.

# Week 4, Part 1 - Neural Networks for Image Classification
Building and training neural networks on the CIFAR-10 dataset.

Learning goals of this homework task:

* Analyse Activation Functions: Compare the convergence speed and stability of non-saturating (ReLU) vs. saturating (Sigmoid, Tanh) activation functions.
* Architecture Design: Understand the trade-offs between network Width and Depth and identify the "Law of Diminishing Returns" when scaling model capacity.
* Hyperparameter Optimisation: Evaluate how learning rate, batch size, and choice of optimiser (Adam, SGD, RMSprop) influence training dynamics and final test performance.
* Generalisation & Regularisation: Implement <BatchNorm> to stabilise gradients and <Dropout> to prevent "memorisation" of training noise, especially in complex datasets like Cat vs. Dog.
* Binary to Multiclass Transition: Scale a classification pipeline from binary (2 classes) to multiclass (10 classes) using CrossEntropyLoss and raw Logits.
* Diagnose Overfitting: Identify the "Confidence Gap" by monitoring the divergence between Training Loss and Test Accuracy/Loss.


Key Technical Takeaways:

* Numerical Stability: Applying BatchNorm before activation functions helps prevent gradient explosion and allows for higher learning rates.
* The Vanishing Gradient Problem: Visualising how Sigmoid/Tanh loss curves "flatline" in early epochs compared to the aggressive learning of ReLU.
* Generalisation Gap: Understanding that a model with high training accuracy but high test loss is "overconfident" and requires stochastic regularisation like Dropout.
* Layer Sequencing: Mastering the standard deep learning "sandwich" layer order: <Linear → BatchNorm → Activation → Dropout.>

# Week 4, Part 2 - Character-Level Language Model
building a character-level RNN language model to generate dinosaur names.


Learning goals of this hometask:

* Understand Character-Level Preprocessing: Implement one-hot encoding for a fixed vocabulary and manage token_to_id mappings for text-to-tensor conversion.
* Master RNN Architecture & Tensor Flow: Visualise how data flows through an LSTM, understanding the transition from 3D sequence blocks [batch, seq, hidden] to 2D flattened logits [batch * seq, vocab].
* Differentiate Hidden vs. Cell States: Understand the dual-memory system of LSTMs (short-term $h$ vs. long-term $c$) and their roles in both training and name generation.
* Implement Robust Training Loops: Learn to handle dynamic batch sizes (the "last batch" problem) and correctly initialise/detach hidden states to prevent memory leaks and unintended backpropagation through history.
* Stabilise Recurrent Gradients: Apply Gradient Clipping to prevent the "exploding gradient" problem common in deep recurrent networks.
* Evaluate Generative Models: Understand why Cross-Entropy Loss is the primary metric for text generation and how to interpret a loss "plateau" as the balance between memorisation and creativity.
* Compare Stochastic Decoding Strategies: Implement and tune Top-K Sampling and Temperature Scaling to control the trade-off between "safe" patterns and creative variation.
* Implement Deterministic Search Algorithms: Build a Beam Search decoder using log-probability math to find the globally most probable sequence, moving beyond simple Greedy Search.

# Week 6  - Build Your Own Tiny Transformer (Language Model)
Learning goals of this hometask:

* Master the Self-Attention Mechanism: Understand how to project input embeddings into Query ($Q$), Key ($K$), and Value ($V$) spaces and use their dot products to create a communication map between tokens.
* Implement Causal Masking: Learn to apply a lower-triangular mask to the attention scores to ensure the model cannot "cheat" by looking at future characters during training.
* Build the Transformer Block: Understand the "Pre-norm" architecture, combining Multi-Head Self-Attention and Position-wise Feed-Forward networks into a single repeatable unit.
* Understand Residual Connections: Implement the "Residual Stream" (x = x + \text{sublayer}(x)) to allow gradients to flow through deep stacks of layers without vanishing.
* Manage Positional Embeddings: Implement a learnable positional embedding table and understand how to "broadcast-add" it to token embeddings so the model understands character order.
* Develop Autoregressive Generation: Implement the "loop" required to generate text one character at a time, specifically managing the cropping of the context to stay within the model's fixed block_size.
* Understand Logit-to-Token Mapping: Learn how to use a linear "head" to project high-dimensional vectors back into vocabulary space and use Softmax sampling to produce creative, non-deterministic text.

# Week 8 - Part 1 - Parameter-Efficient Fine-Tuning (LoRA)
Implementing Low-Rank Adaptation (LoRA) from scratch to fine-tune a pre-trained GPT-2 (124M) on the TinyShakespeare dataset.

Learning goals of this hometask:

* Implement LoRA from Scratch: Build a LoRALinear wrapper to execute the residual update $y = W_0x + \frac{\alpha}{r}BAx$.
* Master Module Injection: Use recursive logic to swap standard layers (c_attn, c_proj) with trainable adapters.
* Calculate Parameter Efficiency: Understand how rank-$r$ decomposition reduces trainable weights to $<1\%$ (e.g., ~0.65% for GPT-2 Small).
* Monitor Catastrophic Forgetting: Use Perplexity (PPL) on a Control Corpus (Jane Austen) to ensure general language capability is preserved during specialisation.
* Optimise Training Dynamics: Implement AdamW over filtered parameters and utilise a Warmup + Cosine Decay learning rate schedule.
* Validate with Industry Tools: Compare hand-rolled PPL results and parameter counts against the Hugging Face peft library.

Key Technical Takeaways:
* Zero-Initialization Strategy: Initializing matrix $B$ with zeros ensures the initial $\Delta W = 0$, preserving pre-trained performance at step 0.
* Optimiser VRAM Savings: Filtering for requires_grad=True prevents AdamW from storing redundant momentums for frozen weights.The Scaling Factor ($\alpha/r$): Understanding how this hyperparameter stabilises the effective learning rate when modifying the adapter rank ($r$).
* Functional Equivalence: Verifying that custom nn.Linear conversions and PEFT's Conv1D wrapping produce identical numerical results. 
***
*Project developed by Mariia Osokina as part of AI Performance Engineering curriculum from Nebius Academy https://academy.nebius.com/ai-engineering-uk.*
