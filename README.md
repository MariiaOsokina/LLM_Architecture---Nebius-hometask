# LLM_Architecture---Nebius-hometask

# Week 1 - Optimization in PyTorch — Gradient Descent, SGD, Numerical Stability, and L1 Regularization

Learning goals of this hometask:
* Understand preprocessing design choices (tokenization provided, fixed vocabulary).
* Implement and train Logistic Regression manually in PyTorch using SGD.
* Explain why numerical stability matters in softmax and log-loss.
* Understand how optimization parameters like learning rate and batch size affect training.
* Understand the effect of L1 regularization and why it encourages sparsity.
* Understanding how an optimization algorithm behaves when the loss function has different shapes.

# Week 4, Part 1 - Neural Networks for Image Classification
Building and training neural networks on the CIFAR-10 dataset.

Learning goals of this hometask:
* Analyze Activation Functions: Compare the convergence speed and stability of non-saturating (ReLU) vs. saturating (Sigmoid, Tanh) activation functions.
* Architecture Design: Understand the trade-offs between network Width and Depth and identify the "Law of Diminishing Returns" when scaling model capacity.
* Hyperparameter Optimization: Evaluate how learning rate, batch size, and choice of optimizer (Adam, SGD, RMSprop) influence training dynamics and final test performance.
* Generalization & Regularization: Implement <BatchNorm> to stabilize gradients and <Dropout> to prevent "memorization" of training noise, especially in complex datasets like Cat vs. Dog.
* Binary to Multiclass Transition: Scale a classification pipeline from binary (2 classes) to multiclass (10 classes) using CrossEntropyLoss and raw Logits.
* Diagnose Overfitting: Identify the "Confidence Gap" by monitoring the divergence between Training Loss and Test Accuracy/Loss.

Key Technical Takeaways:
* Numerical Stability: Applying BatchNorm before activation functions helps prevent gradient explosion and allows for higher learning rates.
* The Vanishing Gradient Problem: Visualizing how Sigmoid/Tanh loss curves "flatline" in early epochs compared to the aggressive learning of ReLU.
* Generalization Gap: Understanding that a model with high training accuracy but high test loss is "overconfident" and requires stochastic regularization like Dropout.
* Layer Sequencing: Mastering the standard deep learning "sandwich" layer order: <Linear → BatchNorm → Activation → Dropout.>

# Week 4, Part 2 - Character-Level Language Model
building a character-level RNN language model to generate dinosaur names.


Learning goals of this hometask:

* Understand Character-Level Preprocessing: Implement one-hot encoding for a fixed vocabulary and manage token_to_id mappings for text-to-tensor conversion.
* Master RNN Architecture & Tensor Flow: Visualize how data flows through an LSTM, understanding the transition from 3D sequence blocks [batch, seq, hidden] to 2D flattened logits [batch * seq, vocab].
* Differentiate Hidden vs. Cell States: Understand the dual-memory system of LSTMs (short-term $h$ vs. long-term $c$) and their roles in both training and name generation.
* Implement Robust Training Loops: Learn to handle dynamic batch sizes (the "last batch" problem) and correctly initialize/detach hidden states to prevent memory leaks and unintended backpropagation through history.
* Stabilize Recurrent Gradients: Apply Gradient Clipping to prevent the "exploding gradient" problem common in deep recurrent networks.
* Evaluate Generative Models: Understand why Cross-Entropy Loss is the primary metric for text generation and how to interpret a loss "plateau" as the balance between memorization and creativity.
* Compare Stochastic Decoding Strategies: Implement and tune Top-K Sampling and Temperature Scaling to control the trade-off between "safe" patterns and creative variation.
* Implement Deterministic Search Algorithms: Build a Beam Search decoder using log-probability math to find the globally most probable sequence, moving beyond simple Greedy Search.


***
*Project developed by Mariia Osokina as part of AI Performance Engineering curriculum from Nebius Academy https://academy.nebius.com/ai-engineering-uk.*
