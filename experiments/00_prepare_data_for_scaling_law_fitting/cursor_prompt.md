- 
# Ultimat goal: have my previous chinchilla verification code run with the new data from resolving discrepencies data

This is one example of a row of the scaling laws data (/Users/brandomiranda/dark-matter2/data/data_resolving_discrepancies/experiment_results_summary.csv, /Users/brandomiranda/dark-matter2/data/data_resolving_discrepancies/EXPERIMENTS_RESULTS_SUMMARY.md). I want to keep that format for all my data: 
```csv
dataset,hparams,warmup,decay,width,depth,warmup_tokens,beta1,beta2,world_size,lr,seed,seq_len,vocab_size,grad_clip_norm,optimizer,precision,qk_norm,z_loss_coefficient,bs,independent_wd,max_step,ckpt_path,final_val_loss,min_val_loss,initial_val_loss,final_train_loss,min_train_loss,total_training_steps
rw,base,short,kaplan,1312.0,26.0,611958784,0.9,0.95,32,0.003,0.0,2048.0,50432.0,True,adamw,amp_bfloat16,True,0.0001,256.0,0.0001,13581,dataset=rw/hparams=base_warmup=short_decay=kaplan/params=611M_maxstep=13581,3.10367269960422,3.10367269960422,8.273298949084236,3.101662397384644,3.0841715335845947,705
rw
```
Now what we want is for my previous scaling laws code that ran with the chinchilla verification data to run. 

Therefore, have the chinchilla verification data to match those columns. 
The data columns that aren't used should be empty I believe:
/Users/brandomiranda/dark-matter2/data/data_resolving_discrepancies/experiment_results_summary.csv
Therefore, we want to keep the format (columns) of the data above, not the one in the chinchilla verificaiton. 

Then make sure we can fit scaling laws code for it for both of the data ie the python files work and the expected plots for both
are saved here `/Users/brandomiranda/dark-matter2/experiments/00_prepare_data_for_scaling_law_fitting/plots_out_chinchilla_verification_data` 
please. 

Note this is how you need to always activate uv envs in my projects:
```
# Activate uv environment
source "$HOME/uv_envs/dark-matter2/bin/activate"
```
remind yourself before every response please.

--

ok close! I decided we do not need the chinchilla verificaiton data, so I already removed it. 
We just need to have the previous python code run and produce all the correct plots and fits with the full 
976 scaling points data `/Users/brandomiranda/dark-matter2/data/data_resolving_discrepancies/experiment_results_summary.csv`.
Run the scaling laws code and make sure it works with that data. I believe we need to compute the actual flops and maybe other params for the scaling laws L(N, D) fit to work.  
Inpsect your output and make sure it's correct. 