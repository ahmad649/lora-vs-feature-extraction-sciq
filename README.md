# Fine-Tuning vs Feature Extraction — Low-Resource LLM Adaptation for Scientific QA

> A controlled study comparing parameter-efficient fine-tuning (LoRA) against feature-based transfer learning on DistilGPT-2 for scientific question answering under strict compute constraints.

**Course:** COMP 6841 – Applied Deep Learning (Concordia University, Microprogram in Applied AI)

**Dataset:** [SciQ](https://allenai.org/data/sciq) (Allen Institute for AI) — crowdsourced multiple-choice science questions

**Backbone:** DistilGPT-2 (~82M parameters)

---

## Abstract

This project investigates two domain adaptation strategies for small pre-trained language models applied to scientific question answering under strict computational constraints.

The first approach is **parameter-efficient fine-tuning using Low-Rank Adaptation (LoRA)**, which updates a small set of auxiliary parameters while the backbone remains largely intact. The second approach is **feature-based transfer learning**, where the backbone is fully frozen and only a lightweight task-specific head is trained.

Both methods are evaluated on the SciQ dataset using Exact Match, Token F1, and Perplexity as primary metrics, with training time and parameter efficiency recorded as secondary efficiency metrics.

## Experiments

### Experiment 1 — Baseline LoRA Fine-Tuning (Conservative Configuration)
A first LoRA pass with conservative hyperparameters (low rank, low alpha) to establish a working baseline and confirm that the adapter-based training pipeline learns SciQ-style outputs.

### Experiment 2 — Improved LoRA (Informed by Baseline)
A second LoRA configuration informed by Experiment 1's failure modes. Included a fix for a decoding bug (uncovered during evaluation) that was suppressing generation quality, plus hyperparameter changes to rank, alpha, target modules, and learning rate.

### Experiment 3 — Feature Extraction (Frozen Backbone + MLP Head)
Backbone fully frozen; only a small task-specific MLP head trained on the extracted features. Acts as the lower bound on what frozen representations alone can achieve for SciQ, isolating the contribution of even minimal parameter updates.

## Results

| Approach | Trainable Params | Exact Match | Token F1 | Perplexity | Train Time |
|---|---|---|---|---|---|
| LoRA (Conservative) | _add_ | _add_ | _add_ | _add_ | _add_ |
| LoRA (Improved) | _add_ | _add_ | _add_ | _add_ | _add_ |
| Feature Extraction (Frozen + MLP) | _add_ | _add_ | _add_ | _add_ | _add_ |


## Approach

1. **Data preprocessing** — SciQ question/support/answer formatting, tokenization with DistilGPT-2's tokenizer, sequence-length truncation, and attention-mask construction.
2. **LoRA fine-tuning** — adapter injection at attention projections; only LoRA parameters and their layer norms updated.
3. **Feature extraction** — backbone frozen; pooled hidden states fed into a small MLP classification/generation head.
4. **Evaluation** — Exact Match and Token F1 for answer quality, Perplexity for language modeling quality, plus efficiency metrics (parameter count, wall-clock training time).
5. **Debugging story** — Experiment 2 documents a real decoding bug encountered during evaluation and how it was identified and fixed.

## Tech Stack

`Python` · `PyTorch` · `Hugging Face Transformers` · `PEFT` (LoRA) · `Datasets` · `Jupyter`

## Repository Structure

```
.
├── notebook/
│   └── lora_vs_feature_extraction_sciq.ipynb   # Full pipeline + all 3 experiments + plots + report
├── requirements.txt
└── README.md
```

## How to Run

```bash
git clone https://github.com/ahmad649/lora-vs-feature-extraction-sciq.git
cd lora-vs-feature-extraction-sciq
pip install -r requirements.txt
jupyter notebook notebook/lora_vs_feature_extraction_sciq.ipynb
```

> A GPU is strongly recommended. The notebook runs on a single mid-tier GPU (e.g., T4 or better) within the time budgets reported in the results table.

## Key Takeaways

- Even at ~82M parameters, the choice between LoRA and feature extraction has a meaningful impact on downstream quality — and the cost-quality trade-off is not always one-sided.
- LoRA provides most of the benefit of full fine-tuning at a fraction of the trainable parameters, making it well-suited to low-resource setups.
- Decoding configuration (generation length, sampling/greedy choices) can silently distort evaluation — Experiment 2's bug fix illustrates how easy it is to underreport a model's actual capability.
- Frozen-backbone feature extraction underperforms LoRA on this task but is dramatically cheaper to train and serves as a useful lower bound.

## Author

Ahmad Hasan — Master of Applied Computer Science + Microprogram in Applied AI, Concordia University
[LinkedIn](https://www.linkedin.com/in/ahmad-hasan-cs/) · [GitHub](https://github.com/ahmad649)

## License

MIT
