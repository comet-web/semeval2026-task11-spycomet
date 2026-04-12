# SpyComet at SemEval-2026 Task 11

System description for SemEval-2026 Task 11 (Subtask 1): Disentangling Content and Formal Reasoning in Language Models.

## System: MLA-CI

MLA-CI (Multi-Layer Adversarial for Content Invariance) is a DeBERTa-v3-base classifier for syllogism validity classification. The system combines:
1. Multi-layer feature extraction (layers 2, 6, -2)
2. Gradient-reversal adversarial training
3. Structure-preserving template augmentation
4. Implausible-class oversampling
5. Focal loss

**Official test results**: 79.06% accuracy, 4.17% content effect, combined score 29.92 

**Key finding**: Adversarial training is counterproductive when template augmentation is present. Removing it improves the validation score from 26.41 ± 0.99 to 38.15 ± 5.32 across three seeds.

## Repository Structure

- `notebooks/` — All Colab notebooks for training and experiments
  - `MLACI5.ipynb` — Submitted system (full pipeline)
  - `ablation_study.ipynb` — Leave-one-out ablation (6 configurations)
  - `optimal_experiment.ipynb` — Aug + MultiLayer only experiment
  - `vanilla_baseline.ipynb` — Vanilla DeBERTa baseline
  - `multi_seed.ipynb` — Multi-seed robustness experiments
- `results/` — JSON files with all experimental results

## Requirements

- Python 3.10+
- PyTorch
- transformers==4.47.1
- accelerate==0.26.1
- scikit-learn
- pandas, numpy, tqdm

All notebooks are designed to run in Google Colab with a T4 GPU.

## How to Reproduce

1. Obtain the task data from the [SemEval-2026 Task 11 organizers](https://sites.google.com/view/semeval2026-task11/)
2. Upload `train_data.json` and `test_data_subtask_1.json` to your Colab environment
3. Run notebooks in this order:
   - `MLACI5.ipynb` (cells 1-11 for data prep, then cell 12+ for training)
   - `ablation_study.ipynb` (requires cells 1-11 from MLACI5 first)
   - Other notebooks follow the same pattern


## Citation
```bibtex
@inproceedings{spycomet-semeval-2026-task11,
    title = "SpyComet at SemEval-2026 Task 11: When Adversarial Debiasing Backfires---Comparing Data-Level and Model-Level Strategies for Content-Invariant Syllogistic Reasoning",
    booktitle = "Proceedings of the 20th International Workshop on Semantic Evaluation (SemEval-2026)",
    year = "2026",
}
```

## License

MIT
