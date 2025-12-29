# 🧬 Mistral 7B Finetune: GCSE Biology Automated Marker

An end-to-end LLM fine-tuning pipeline designed to grade GCSE Biology answers, assign marks based on strict schemes, and provide constructive, teacher-style feedback.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Transformers-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📖 Project Overview
Teachers spend hundreds of hours grading assessments. This project fine-tunes a **Mistral 7B Instruct** model to act as an automated examiner. It doesn't just say "Correct" or "Incorrect"—it explains *why* a student lost marks and provides a revision tip, mimicking a real human tutor.

**Key Technical Challenges Solved:**
* **Memory Optimization:** Implemented **4-bit Quantization (QLoRA)** to fine-tune a 7B parameter model on consumer hardware (T4/A100 GPUs).
* **Data Engineering:** Built a custom pipeline to convert raw CSV grading logs into **ChatML formatted JSONL** datasets.
* **Efficient Inference:** Optimized the inference pipeline to run on free-tier Google Colab GPUs (15GB VRAM limit).

## 📂 Repository Structure

| File | Description |
| :--- | :--- |
| `01_mistral_finetune_pipeline.ipynb` | **The Training Loop.** Handles data preprocessing, 4-bit loading, QLoRA adapter configuration, and the training process. |
| `02_inference_demo.ipynb` | **The Inference Engine.** Loads the fine-tuned adapter from storage and runs the grading bot on new, unseen student answers. |
| `formatted_biology_chatml.jsonl` | The cleaned dataset used for training (converted from CSV). |

## 🛠️ Tech Stack
* **Model:** `mistralai/Mistral-7B-Instruct-v0.2`
* **Libraries:** `transformers`, `peft`, `bitsandbytes`, `trl`, `pandas`
* **Techniques:** LoRA (Low-Rank Adaptation), QLoRA (Quantization), Supervised Fine-Tuning (SFT)
* **Infrastructure:** Google Colab Pro (A100 for training, T4 for inference)

## 🚀 How to Run

### Prerequisites
* A Hugging Face Account & Access Token (Write permissions).
* Google Colab (Free tier works for inference; Pro recommended for training).

### Step 1: Training (`01_mistral_finetune_pipeline.ipynb`)
1.  Open the notebook in Google Colab.
2.  Upload your dataset or generate it using the built-in script.
3.  Add your Hugging Face token to Colab Secrets as `HF_TOKEN`.
4.  Run all cells to fine-tune the model.
5.  The adapter weights will be saved to your Google Drive.

### Step 2: Inference (`02_inference_demo.ipynb`)
1.  Open the notebook.
2.  Mount Google Drive (to access your saved adapter).
3.  Run the inference block.
4.  Input a student answer to see the AI grade it in real-time.

## 📊 Data Format
The model is trained on specific prompt pairs formatted in **ChatML**:

**User Input:**
```text
Question: Explain why muscle cells have more mitochondria than skin cells.
Max Marks: 3
Student Answer: Because they are big.