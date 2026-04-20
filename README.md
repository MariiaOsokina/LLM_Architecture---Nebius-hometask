# LLM_Architecture---Nebius-hometask

# Week 1 - Optimization in PyTorch — Gradient Descent, SGD, Numerical Stability, and L1 Regularization

Learning goals of this hometask
By completing this assignment:
* Understand preprocessing design choices (tokenization provided, fixed vocabulary).
* Implement and train Logistic Regression manually in PyTorch using SGD.
* Explain why numerical stability matters in softmax and log-loss.
* Understand how optimization parameters like learning rate and batch size affect training.
* Understand the effect of L1 regularization and why it encourages sparsity.
* Understanding how an optimization algorithm behaves when the loss function has different shapes.


# Week 4, Part 1 - CIFAR-10 Classification: A Deep Learning Journey

This repository documents a systematic approach to optimizing neural networks for image classification using **PyTorch**. Through a series of experimental "Battles," this project scales from simple binary classification to a robust, regularized multiclass model capable of identifying all 10 CIFAR-10 categories.

## 🛠 Technical Stack
* **Framework:** PyTorch
* **Optimization:** Adam, SGD + Momentum, RMSprop
* **Regularization:** Batch Normalization, Dropout
* **Data:** CIFAR-10 (32x32 RGB images)

## 🧪 Phase 1: Binary Classification Optimization

### Task 1.4: Frog vs. Ship (Optimization Benchmarking)
The goal was to find the most efficient combination of hyperparameters to solve a relatively simple classification task.

* **The Activation Battle:** **ReLU** outperformed Sigmoid and Tanh by eliminating the vanishing gradient problem, allowing the model to hit 90%+ accuracy almost instantly.
* **The Architecture Battle:** While "Wide" models had raw power, the **Baseline [128, 64]** was selected as the "Best Practical Model" for its superior capacity-to-stability ratio.
* **The Optimizer Battle:** **Adam** was chosen for final implementation due to its rapid convergence and stability, though **SGD + Momentum** achieved the highest technical peak (94.80%).

### Task 1.5: Cat vs. Dog (The Regularization Battle)
This task proved significantly harder due to the visual similarity of the classes. 

* **The Overfitting Trap:** BatchNorm used alone reached 87.3% training accuracy but crashed on test data (59.4%).
* **The Synergy Winner:** The combination of **BatchNorm + Dropout (0.3)** was the only setup to clear the **>0.64 accuracy** benchmark (Final: **65.55%**), proving that regularization is mandatory for complex features.

## 🚀 Phase 2: Multiclass Mastery (10 Classes)

### Task 3.4: Final Multiclass Optimization
The final challenge was to achieve **>0.53 accuracy** across all 10 CIFAR-10 classes.

#### **1. Architecture: Depth vs. Width**
To distinguish between 10 complex classes, I scaled the architecture to a 3-layer deep network: **[512, 256, 128]**.
* **Finding:** Depth allowed for better feature abstraction (associative memory), while the width provided the capacity to handle the increased variety of labels. 
* **Result:** This architecture reached a final test accuracy of **57.22%**.

#### **2. Activation Evolution & Convergence**

Benchmarking the 3-layer network against different activations revealed:
* **ReLU:** The efficiency leader, achieving target loss in roughly half the epochs of other functions.
* **Sigmoid:** Suffered from flat gradients in early stages, struggling to propagate error signals through the deeper layers.
* **Tanh:** Showed moderate speed but exhibited "jitter" and instability in later training stages.

#### **3. Critical Hyperparameters**
* **Learning Rate (LR):** Identified **0.001 (Adam)** as the "Goldilocks" setting. 
* **BatchNorm:** Crucial for stabilizing the 3-layer deep structure and preventing gradient explosion.
* **Loss Function:** Utilized **CrossEntropyLoss** paired with raw **Logits** as the output layer, ensuring mathematical consistency with PyTorch’s optimization engine.

---

## 📈 Key Findings & "Interesting Behavior"

* **The Confidence Gap:** I observed instances where Test Accuracy remained flat while Test Loss rose. This indicated the model was becoming "overconfident" in wrong answers—a precursor to overfitting managed by the addition of Dropout.
* **The Law of Diminishing Returns:** Doubling the number of neurons did not double the accuracy. Instead, it increased the **Generalization Gap** (Training 68% vs Test 57%), highlighting the importance of regularization over raw size.
* **Data Preprocessing:** Standardizing inputs in the Dataset class was essential for the weights to converge within the 30-epoch limit.



---

## 🏁 Final "Champion" Configuration

| Component | Choice | Rationale |
| :--- | :--- | :--- |
| **Activation** | **ReLU** | Fastest convergence; non-saturating. |
| **Architecture** | **[512, 256, 128]** | Balanced capacity for 10-class complexity. |
| **Regularization** | **BatchNorm + Dropout** | Prevents memorization of training noise. |
| **Optimizer** | **Adam (0.001)** | Superior speed and adaptive step-sizing. |
| **Output Layer** | **Logits** | Optimized for CrossEntropyLoss. |

---

## 💻 How to Run
1.  Open the provided `.ipynb` notebook in Google Colab.
2.  Set the runtime to **GPU**.
3.  Run the **Setup Block** to download CIFAR-10 and initialize the training/testing loaders.
4.  Execute the **Task 3.4 Block** to see the final 57.22% model in action.

***
*Project developed by Maria Osokina as part of the LLM Architectures curriculum.*
