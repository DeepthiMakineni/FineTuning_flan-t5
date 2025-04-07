Keyword Extraction using Fine-Tuned FLAN-T5

This project demonstrates how to fine-tune the `google/flan-t5-base` model for domain-specific keyword extraction using a `skincare` dataset. The notebook includes data upload, preprocessing, fine-tuning, and evaluation of keyword extraction performance.


## Project Objective

To build a keyword extraction pipeline that identifies important terms and phrases from unstructured skincare-related text, improving performance over generic pre-trained models.

## 📁 File Structure

- `finetuning_google-flan_t5.ipynb` - Main notebook that handles model loading, fine-tuning, and keyword extraction.
- `skincare.txt` - Input text file used for generating training prompts.
- `keywords.jsonl` - Training dataset (manually labeled) used for fine-tuning.
- `README.md` - This file.


## Requirements

```bash
pip install transformers accelerate sentencepiece torch
```

---

## How it Works

### 1. Upload and Read File

```python
uploaded = files.upload()
content = file.read()
```

### 2. Pre-trained Inference (Before Fine-tuning)

```python
Extracted Keywords: vitamin C, niacinamide, retinol, reduces fine lines, stimulates collagen, etc.
```

This shows the base model outputs generic but relevant keywords.

### 3. Fine-tuning the Model

```python
model = SFTTrainer(...)  # with training dataset: skincare → keywords
```

Using Hugging Face `Trainer` and custom JSONL data.

### 4. Post Fine-Tuning Inference

```python
Extracted Keywords: vitamin C, niacinamide, retinol, fine lines, collagen boost, elasticity, dark spot corrector, wrinkle reduction, glow, peptide complex, sunscreen, texture refining, hydration, hyaluronic acid
```

This output shows significant improvement in **keyword count** and **relevance**.

---

## Results

| Metric              | Base Model           | Fine-tuned Model        |
|---------------------|----------------------|--------------------------|
| Keyword Coverage    | Low (5-7 keywords)    | High (12-15 keywords)    |
| Domain Relevance    | Generic               | Skincare-specific        |
| Redundancy          | Medium                | Low                      |
| Precision           | Moderate              | High                     |

---

## 📈 Improvements Made

- Prompt engineered to extract up to 15–30 keywords.
- Customized training data with domain-specific annotations.
- Output formatting improved for clarity and separation.

## Notebook Link

[Open in Google Colab](https://colab.research.google.com/drive/1Fj-pVzk0N1yxTdZ75sNJ-DTKaloNDEyO)
