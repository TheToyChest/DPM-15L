# DPM-15L
Look Mom,  No Softmax! Replaces Softmax with Decoupled Polynomial Mapping in Self Attention and Loss determination. Code and evaluation scripts for DPM-15L, a 25M parameter No-Softmax transformer. Next Token Accuracy examples over 44%.

#LICENSE
Commercial and Non-Commercial Research & Development License
Copyright (c), 2026, Submitting Author(s). All rights reserved.

You are free to use the software to engage in all forms of Research and Development, commercial and non-commercial, including for academic and non-profit purposes and internal commercial purposes ("Purposes"). Purposes does not include commercial exploitation. Licensor hereby grants to the Licensee a worldwide, non-exclusive, non-transferable, royalty-free, revocable license to reproduce, modify, create derivative works from, and use the Software for the Purposes.  No other purposes are permitted.  To the extent patent rights exist or are obtained, no patent rights are granted beyond the Purposes.
Disclaimer of Warranty THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE LICENSOR BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
This License and the rights granted hereunder will terminate automatically if the Licensee fails to comply with any terms herein.

# TheToyChest: DPM-15L 

**A 25M parameter causal language model utilizing Dual-Polynomial Attention (DPA) and true mathematical sparsity.**

This repository contains the model architecture, training configurations, and zero-shot evaluation scripts for **DPM-15L**. By completely abandoning the standard Softmax attention bottleneck and replacing it with a zero-floored pure `ReLU` activation, this architecture achieves strict rank-order mathematical sparsity while maintaining stable gradient flow via a learned polynomial loss landscape.

## 🚀 Quick Start: Zero-Shot Evaluation

You can reproduce our Wikitext-103 zero-shot perplexity and exact-match accuracy without needing local GPU hardware. 

We have provided a standalone Google Colab notebook (`DPM_Wikitext_ZeroShot_Eval.ipynb`) that automatically:
1. Downloads the pre-trained weights from Hugging Face.
2. Tokenizes the Wikitext-103 validation split.
3. Evaluates the strict rank-order Exact-Match Accuracy and NLL Perplexity.

**Model Weights:** Hosted on Hugging Face at [TheToyChest/DPM-15L](https://huggingface.co/TheToyChest/DPM-15L).

*Note: The current champion weights utilize pure ReLU. When evaluating these weights locally, ensure `Config.continuous_abla = False`.*

## 📊 Empirical Proof: The Pareto Frontier
**A Note on Validation Loss:** Because DPM and standard Softmax operate on fundamentally different mathematical loss frameworks (a zero-floored polynomial landscape vs. a dense exponential fraction), their raw NLL Validation Loss values cannot be directly mapped 1:1 against each other. To provide a rigorous, objective comparison, **Next-Token Accuracy** and **Wikitext-103 Zero-Shot Generalization** serve as the framework-agnostic benchmarks used to rate and compare model performance.

| Model Config | Seed | Batch Size | OWT Val Loss | OWT Next-Token Acc | Wiki-103 Perplexity | Wiki-103 Zero-Shot Acc | HellaSwag |
| :---         | :---: | :---:     | :---:         | :---:             | :---:               | :---:                   | :---: |
| **Softmax Baseline** | 42/105 | 32 | 4.05       | 31.14%             | 178.44               | 23.72%                 | 25.22% |
| **DPM (ReLU)** | 42 | 32         | 3.17         | 40.01%             | 81.28               | 28.30%                 | 25.12% |
| **DPM (ReLU)** | 105 | 16         | 2.81         | 43.76%             | 59.81               | 33.66%                 | 25.19% |
| **DPM (ReLU)** | 42 | 16         | 2.71         | 44.81%             | 44.67               | 34.17%                 | 25.67% |
| **DPM (ReLU) 🏆** | 999 | 16     | **2.69**      | **44.26%**       | **44.05**             | **33.15%**           | **25.85%** |

## 🏗️ Architecture Specifications

* **Topology:** 15 Layers, 4 Attention Heads
* **Dimensionality:** 256 Embedding Dim, 1024 FFN Dim
* **Context Window:** 512 Tokens
* **Positional Embeddings:** RoPE (Rotary Position Embedding)
* **Normalization:** RMSNorm (Post-Norm Configuration + Q/K Normalization)
* **Attention Activation:** Pure ReLU 
* **Parameter Count:** 25 Million
* **Training Data:** 1 Epoch on OpenWebText

## 📂 Repository Contents

* `DPM_Wikitext_ZeroShot_Eval.ipynb`: Automated Google Colab script for reproducing zero-shot evaluations.
* *(Add other filenames here as you push them, e.g., `train.py`, `model.py`)*

