# Data sources

## What’s here
0. Original Data: `data_resolving_discrepancies/experiment_results.pickle.xz`  
  - Contains per-run time-series losses (pandas DataFrame with `val/loss` and `train/loss` series indexed by step).
1. Scaling Laws Data 1: `data_resolving_discrepancies/scaling_law_data.csv`  
  - 1 CSV; Flattened table of 975 runs/rows/data points 
  - columns: `dataset,hparams,warmup,decay,width,depth,num_params,batch_size,lr,max_step,tokens,flops,final_val_loss,final_train_loss,final_step,ckpt_path` (and derived `num_params`).
2. Scaling Laws Data 2: `data_hitchhickers_guide/*.csv`  
  - 65 CSVs; ~572 columns per file; ~1,639,962 rows total (row count from `hitchhickers_guide_unified.csv`)
  - key columns: `tokens_seen` (training tokens), `data` (dataset name), `model_name`, `model_type`, `scaled_set`, many eval metrics, and a `loss` column. 
  
  includes many eval metrics plus a `loss` column.
- Unified, loss-only CSVs (trimmed to a common schema and finite numeric `loss`):  
  - `data_resolving_discrepancies/scaling_law_data_unified.csv` (975 rows)  
  - `data_hitchhickers_guide/hitchhickers_guide_unified.csv` (1,639,961 rows)  
  - `scaling_law_unified.csv` (combined 1,640,936 rows)

## Unified schema (columns kept)
`num_params, tokens, flops, loss, dataset, model_name, source, family, arch`
- Transform notes: `tokens_seen→tokens`, `data→dataset`, `final_val_loss→loss`, `ckpt_path→model_name`; add `source` tag; drop rows with non-numeric/NaN `loss`.

## Totals to keep in mind
- Hitchhiker raw (all CSVs, before filtering): 7,021,204 lines (includes 65 headers)
- Hitchhiker unified (filtered, single header): 1,639,962 lines
- Resolving Discrepancies unified: 976 lines
- Combined unified: 1,640,937 lines
- Arithmetic check: 1,639,961 Hitchhiker data rows + 975 Resolving data rows = 1,640,936 data rows; plus one combined header = 1,640,937 lines
