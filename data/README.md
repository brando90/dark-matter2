# Data sources

## What’s here

## Ludgwig's data: Resolving Discrepancies of Compute Optimal Scaling Laws

0. Original Data: `data_resolving_discrepancies/experiment_results.pickle.xz`  
  - Contains per-run time-series losses (pandas DataFrame with `val/loss` and `train/loss` series indexed by step).

1. Scaling Laws Data (full): `data_resolving_discrepancies/scaling_law_data_full.csv` (`.csv.gz` also present)  
  - 975 rows, 36 columns.  
  - Includes all upstream columns; time-series columns (`val/loss`, `val/loss_std`, `train/loss`, `train/batch_time`, `train/data_time`, `train/lr`) are JSON strings keyed by training step.  
  - Adds derived helpers: `dataset_readable`, `num_params`, `tokens`, `flops`, `final_val_loss`, `final_train_loss`, `final_step`.

2. Scaling Laws Data (slim): `data_resolving_discrepancies/scaling_law_data.csv`  
  - 975 rows; 21 columns; flattened summary for scaling-law fitting (no time-series).  
  - columns: `dataset,hparams,warmup,decay,width,depth,seq_len,vocab_size,num_params,batch_size,lr,max_step,tokens,flops,final_val_loss,final_train_loss,final_step,optimizer,precision,seed,ckpt_path`.

Summary:
1. Number of scaling law's data points: TODO
2. Number of columns/rows/features: TODO

References:
- [Github, resolving scaling law discrepdancies](https://github.com/formll/resolving-scaling-law-discrepancies)
- [Arxiv, resolving scaling law discrepdancies](https://arxiv.org/abs/2406.19146)

## Hitchhicker's guide to scaling laws

2. Scaling Laws Data 2: `data_hitchhickers_guide/*.csv`  
  - 65 CSVs; ~572 columns per file; ~1,639,962 rows total (row count from `hitchhickers_guide_unified.csv`)
  - key columns: `tokens_seen` (training tokens), `data` (dataset name), `model_name`, `model_type`, `scaled_set`, many eval metrics, and a `loss` column. 
  
  includes many eval metrics plus a `loss` column.
- Unified, loss-only CSVs (trimmed to a common schema and finite numeric `loss`):  
  - `data_resolving_discrepancies/scaling_law_data_unified.csv` (975 rows)  
  - `data_hitchhickers_guide/hitchhickers_guide_unified.csv` (1,639,961 rows)  
  - `scaling_law_unified.csv` (combined 1,640,936 rows)

## Unified schema (columns kept)
 TODO

## Totals to keep in mind
- Hitchhiker raw (all CSVs, before filtering): 7,021,204 lines (includes 65 headers)
- Hitchhiker unified (filtered, single header): 1,639,962 lines
- Resolving Discrepancies unified: 976 lines
- Combined unified: 1,640,937 lines
- Arithmetic check: 1,639,961 Hitchhiker data rows + 975 Resolving data rows = 1,640,936 data rows; plus one combined header = 1,640,937 lines
