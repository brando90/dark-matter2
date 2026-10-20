# DM-4 GPU-hour estimate (L40S, FLOP-based; phase 0 replaces the ASSUME lines with measured throughput)

 phase  width  N_nonemb  runs  tokens_B  gpuh_pess  gpuh_central  gpuh_opt  gpuh_measured
     1    128   1048576    41      4.02       1.82          1.09      0.73           5.88
     1    160   1740800    36      4.20       2.59          1.56      1.04           6.24
     1    224   4300800    41      8.36       8.79          5.27      3.52          13.49
     1    288   7962624    36      9.63       7.79          5.19      3.74          17.66
     1    320  12533760    36     12.28      13.08          8.72      6.28          24.56
     2    384  17694720     4      1.95       2.70          1.80      1.29           4.16
     2    480  33177600     4      3.01       4.03          2.88      2.31           8.20
     2    576  55738368     4      4.45       9.03          6.45      5.16          16.40
     2    640  76185600     4      5.69      14.87         10.62      8.50          24.06
     2    704 113541120     4      7.82      28.61         20.43     16.35          46.18

       gpuh_pess  gpuh_central  gpuh_opt  gpuh_measured
phase                                                  
1           34.1          21.8      15.3           67.8
2           59.2          42.2      33.6           99.0

Total runs: 210 (phase 1: 190, phase 2: 20)
Total GPU-h: pessimistic 93.3, central 64.0, optimistic 48.9
Largest single run (central): 8.58 GPU-h ({'width': 704, 'vocab': 50432, 'tie': 'untied', 'run': 'const'})

MEASURED (phase 0 probe on one NVIDIA L40S, bytes per token measured): total 166.8 GPU-h; phase 1 67.8, phase 2 99.0; largest single run 20.77 GPU-h ({'width': 704, 'vocab': 50432, 'tie': 'untied', 'run': 'const'})
Not yet probed, scaled from the widest probed body by FLOPs per token: widths [704] (46.2 GPU-h of the total)
