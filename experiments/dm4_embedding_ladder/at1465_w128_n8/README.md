# AT-1465: w128 tied vs untied, seeds 3-7 added (n=8 per cell)
Launched Oct 26 18:09 on mercury1 GPUs 0-8 (RTX A4000), 30 const-only runs (bpe8k, bpe32k, neox x tied/untied x seeds 3-7), `--no-cooldown`, same lr 0.0115, bs 28, beta2 0.99, warmup 7503872 tokens as the frozen rows. All 30 finished by 20:02. Raw result.json files: mercury1 /lfs/mercury1/0/brando9/at1211/dm4data/ext_seeds/. See dm4_w128_n8_summary.txt.
Reading: untied beats tied at 8k vocab (-0.0027 nats/byte, se 0.0012, 6/8 seeds); tied beats untied at 32k (+0.0016, se 0.0014, 5/8) and NeoX-50k (+0.0029, se 0.0013, 5/8). Exploratory, w128 only.

## w160 d4 frozen rows (seeds 0-2, mercury2 A4000s, n=3 per cell, same metric)
vocab 8192:  tied 1.1171 untied 1.1147 diff -0.0024 sd 0.0038 untied better 2/3
vocab 32768: tied 1.0717 untied 1.0731 diff +0.0014 sd 0.0024 untied better 1/3
vocab 50432: tied 1.0690 untied 1.0713 diff +0.0023 sd 0.0007 untied better 0/3
Same sign pattern as w128. w224 rows running on mercury2 since Oct 26 20:54 (~05:00 Oct 27).
