# DPM-15L
Look Mom,  No Softmax! Replaces Softmax with Decoupled Polynomial Mapping in Self Attention and Loss determination. Code and evaluation scripts for DPM-15L, a 25M parameter No-Softmax transformer. Next Token Accuracy examples over 44%.

Decoupled Polynomial Mapping (DPM) architecture replaces standard exponential Softmax operations in self-attention and loss computation in a language model transformer.  DPM implements a learnable multi-term polynomial, $a \cdot r^2 + b \cdot r^z$ (where $r$ represents the pre-activation scores). By optimizing the coefficients $a$, $b$, and $z$ during training, DPM enables layer-specific learned activation curves. Evaluating a 25M-parameter DPM transformer on WikiText-103 demonstrates a substantial reduction in validation loss, achieving a 10-run average perplexity of 15.61 compared to a 35.78 ReLU-squared baseline. Correlation studies confirm this improved validation loss efficiently maps to accelerated next-token prediction accuracy. We address initial routing vulnerabilities observed during lexical perturbation by introducing a 1.5\% token swap ablation, which restores robust routing stability. DPM's scalability and zero-shot capabilities are further validated by a single-epoch WebText corpus training run, achieving 70.5\% overall syntactic accuracy across 67,000 BLiMP evaluation pairs.

# TheToyChest: DPM-15L 

**A 25M parameter causal language model utilizing Decoupled Polynomial Mapping (DPM) and true mathematical sparsity.**

This repository contains the model architecture, training configurations, and zero-shot evaluation scripts for a 15 layer DPM language model transformer. This architecture removes Softmax in self attention and in loss determination and replaces it with DPM, a zero-floored `ReLU` activation and learned polynomial expansion.

## 🚀 Quick Start: Zero-Shot Evaluation

You can reproduce our Wikitext-103 zero-shot perplexity and exact-match accuracy without needing local GPU hardware. 

We have provided a standalone Google Colab notebook (`DPM_Wikitext_ZeroShot_Eval.ipynb`) that automatically:
1. Downloads the pre-trained weights from Hugging Face.
2. Tokenizes the Wikitext-103 validation split.
3. Evaluates the strict rank-order Exact-Match Accuracy and NLL Perplexity.

**Model Weights:** Hosted on Hugging Face at [TheToyChest/DPM-15L](https://huggingface.co/TheToyChest/DPM-15L).

## 📊 Empirical Proof: The Pareto Frontier
**A Note on Validation Loss:** DPM utilizes ReLU, a rectifying function, and zeros out some signals. This alters the calculation for NLL Validation Loss compared to Softmax, which does not zero out any signals.  Because DPM and standard Softmax operate on different mathematical loss frameworks (a zero-floored polynomial  vs. a dense exponential ), their raw NLL Validation Loss values cannot be directly mapped 1:1 against each other without further context. To provide an objective comparison, **Next-Token Accuracy** and **Wikitext-103 Zero-Shot Generalization** results are used as objective benchmarks to compare model performance.

| Model Config | Seed | Batch Size | OWT Val Loss | OWT Next-Token Acc | Wiki-103 Perplexity | Wiki-103 Zero-Shot Acc | HellaSwag |
| :---         | :---: | :---:     | :---:         | :---:             | :---:               | :---:                   | :---: |
| **Softmax Baseline** | 105 | 16 | 4.04       | 30.88%             | 179.38               | 23.47%                 | 25.52% |
| **DPM (ReLU)** | 42 | 32         | 3.17         | 40.01%             | 81.28               | 28.30%                 | 25.12% |
| **DPM (ReLU)** | 105 | 16         | 2.81         | 43.76%             | 59.81               | 33.66%                 | 25.19% |
| **DPM (ReLU)** | 42 | 16         | 2.71         | 44.81%             | 44.67               | 34.17%                 | 25.67% |
| **DPM (ReLU) ** | 999 | 16     | 2.69      | 44.26%       | 44.05             | 33.15%           | 25.85% |

## 🏗️ Architecture Specifications

* **Topology:** 15 Layers, 4 Attention Heads
* **Dimensionality:** 256 Embedding Dim, 1024 FFN Dim
* **Context Window:** 512 Tokens
* **Positional Embeddings:** RoPE (Rotary Position Embedding)
* **Normalization:** RMSNorm (Post-Norm Configuration + Q/K Normalization)
* **Attention Activation:** Pure ReLU and learned polynomial expansion 
* **Parameter Count:** 25 Million
* **Training Data:** 1 Epoch on OpenWebText

## 📂 Repository Contents

* `DPM_Wikitext_ZeroShot_Eval.ipynb`: This is an automated Google Colab script for reproducing zero-shot evaluations.

* #LICENSE
Commercial and Non-Commercial Research & Development License
Copyright (c), 2026, Submitting Author(s). All rights reserved.

You are free to use the software to engage in all forms of Research and Development, commercial and non-commercial, including for academic and non-profit purposes and internal commercial purposes ("Purposes"). Purposes does not include commercial exploitation. Licensor hereby grants to the Licensee a worldwide, non-exclusive, non-transferable, royalty-free, revocable license to reproduce, modify, create derivative works from, and use the Software for the Purposes.  No other purposes are permitted.  To the extent patent rights exist or are obtained, no patent rights are granted beyond the Purposes.
Disclaimer of Warranty THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE LICENSOR BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
This License and the rights granted hereunder will terminate automatically if the Licensee fails to comply with any terms herein.


