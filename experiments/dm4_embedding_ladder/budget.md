# DM-4 GPU-hour estimate (L40S, FLOP-based; phase 0 replaces the ASSUME lines with measured throughput)

 phase  width  N_nonemb  runs  tokens_B  gpuh_pess  gpuh_central  gpuh_opt
     1    128   1048576    41      4.18       1.87          1.12      0.75
     1    160   1740800    36      4.37       2.67          1.60      1.07
     1    224   4300800    41      8.69       9.05          5.43      3.62
     1    288   7962624    36     10.03       8.05          5.37      3.86
     1    320  12533760    36     12.78      13.53          9.02      6.50
     2    384  17694720     4      2.03       2.79          1.86      1.34
     2    480  33177600     4      3.14       4.18          2.98      2.39
     2    576  55738368     4      4.64       9.36          6.68      5.35
     2    640  76185600     4      5.93      15.43         11.02      8.82
     2    704 113541120     4      8.15      29.71         21.22     16.98

       gpuh_pess  gpuh_central  gpuh_opt
phase                                   
1           35.2          22.5      15.8
2           61.5          43.8      34.9

Total runs: 210 (phase 1: 190, phase 2: 20)
Total GPU-h: pessimistic 96.6, central 66.3, optimistic 50.7
Largest single run (central): 9.11 GPU-h ({'width': 704, 'vocab': 8192, 'tie': 'tied', 'run': 'const'})
