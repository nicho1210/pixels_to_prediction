# Pixels to Predictions

Final project for ECE-GY 7123 Deep Learning.

## Task

This project solves a scientific multiple-choice vision reasoning task using SmolVLM-500M-Instruct.

## Main Result

- Zero-shot validation accuracy: 0.6498
- Best LoRA validation accuracy: 0.8025
- Final method: LoRA fine-tuning with best-checkpoint selection

## Repository Structure

- `Final/final_yh6142.pdf`: final report
- `Final/starter_notebook_final.ipynb`: final training and evaluation notebook
- `Final/backup/`: earlier experiment notebooks and backup submissions
- `Final/submission_new_lora.csv`: LoRA submission file
- `data/`: train, validation, test CSV files and images

## Model

Base model: `HuggingFaceTB/SmolVLM-500M-Instruct`

LoRA setting:

- rank: 8
- alpha: 16
- dropout: 0.05
- target modules: q_proj, k_proj, v_proj, o_proj
- batch size: 2
- best epoch: 4
