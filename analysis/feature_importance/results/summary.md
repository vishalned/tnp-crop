# Grouped permutation importance: summary

Δ NRMSE = NRMSE(permuted) − NRMSE(baseline), mean over 3 permutation seeds; CY-Bench: mean over the 7 countries (DE FR IT ES RO HU PL), simulated: mean over tables. Single-variable groups only unless stated.

## Top 3 groups, CY-Bench

- **Default TabICL**: fpar (+0.0056), static (+0.0039), cum_fpar (+0.0019)
- **SCM control**: fpar (+0.0063), static (+0.0057), tmin (+0.0029)
- **WOFOST (dense)**: fpar (+0.0085), cum_fpar (+0.0029), cum_gdd (+0.0019)

## Top 3 groups, Simulated tables

- **Default TabICL**: fpar (+0.0275), rsm (+0.0162), ssm (+0.0053)
- **SCM control**: rsm (+0.0186), fpar (+0.0173), ssm (+0.0066)
- **WOFOST (dense)**: fpar (+0.0374), rsm (+0.0123), cum_gdd (+0.0087)

## Simulated importance far above CY-Bench (>3x and >0.01 absolute), single groups

- **Default TabICL**: fpar (sim +0.0275 vs CY-Bench +0.0056), rsm (sim +0.0162 vs CY-Bench -0.0002)
- **SCM control**: rsm (sim +0.0186 vs CY-Bench +0.0009)
- **WOFOST (dense)**: fpar (sim +0.0374 vs CY-Bench +0.0085), rsm (sim +0.0123 vs CY-Bench +0.0000)

## Share of the total importance in the biggest group (sum of positive mean Δ NRMSE, single groups)

- CY-Bench, Default TabICL: fpar = 34% of the total
- CY-Bench, SCM control: fpar = 24% of the total
- CY-Bench, WOFOST (dense): fpar = 50% of the total
- Simulated, Default TabICL: fpar = 45% of the total
- Simulated, SCM control: rsm = 38% of the total
- Simulated, WOFOST (dense): fpar = 51% of the total

## CY-Bench: all groups (mean over countries / weighted by region-years, Δ anomaly r)

```
                             delta_nrmse_mean                 delta_nrmse_weighted                 delta_anomaly_r_mean                
model                          default_tabicl     scm  wofost       default_tabicl     scm  wofost       default_tabicl     scm  wofost
group_view group                                                                                                                       
family     fam_fpar                    0.0102  0.0113  0.0138               0.0112  0.0127  0.0100              -0.0575 -0.0606 -0.0704
           fam_precip                 -0.0006  0.0018  0.0016              -0.0015  0.0002  0.0009              -0.0296 -0.0391 -0.0368
           fam_temp_gdd               -0.0089 -0.0051 -0.0046              -0.0071 -0.0056 -0.0040               0.0578  0.0181  0.0222
           fam_water_balance          -0.0011  0.0013 -0.0001              -0.0013  0.0006 -0.0011              -0.0181 -0.0256 -0.0296
single     cum_cwb                    -0.0021  0.0003 -0.0002              -0.0018  0.0008 -0.0013              -0.0017 -0.0137 -0.0190
           cum_fpar                    0.0019  0.0028  0.0029               0.0023  0.0029  0.0024              -0.0177 -0.0233 -0.0182
           cum_gdd                    -0.0012  0.0021  0.0019              -0.0005  0.0012  0.0014               0.0011 -0.0124 -0.0149
           cum_prec                    0.0004  0.0013  0.0014               0.0004  0.0008  0.0011              -0.0079 -0.0094 -0.0184
           cwb                         0.0012  0.0014 -0.0006               0.0007  0.0000  0.0000              -0.0095 -0.0051 -0.0062
           et0                        -0.0009 -0.0012 -0.0003              -0.0012 -0.0019  0.0007               0.0022  0.0056 -0.0209
           fpar                        0.0056  0.0063  0.0085               0.0059  0.0065  0.0054              -0.0431 -0.0451 -0.0578
           gdd                        -0.0026 -0.0008 -0.0003              -0.0022 -0.0017  0.0008               0.0126 -0.0049 -0.0027
           prec                       -0.0021  0.0008 -0.0003              -0.0017  0.0003  0.0003              -0.0198 -0.0318 -0.0168
           rad                        -0.0036 -0.0022 -0.0020              -0.0034 -0.0020  0.0001               0.0169  0.0069  0.0092
           rsm                        -0.0002  0.0009  0.0000               0.0007  0.0015  0.0004               0.0011  0.0011  0.0053
           ssm                        -0.0011 -0.0001 -0.0018              -0.0006 -0.0001 -0.0005               0.0149  0.0144  0.0020
           static                      0.0039  0.0057  0.0011               0.0071  0.0086  0.0066              -0.0056 -0.0034  0.0019
           tavg                       -0.0029 -0.0024 -0.0034              -0.0032 -0.0034 -0.0030               0.0226  0.0142  0.0209
           tmax                        0.0019  0.0014  0.0010               0.0021  0.0005  0.0011               0.0003 -0.0029 -0.0038
           tmin                        0.0018  0.0029 -0.0030               0.0011  0.0006 -0.0017              -0.0115 -0.0139  0.0041
```

## Baseline vs the existing exp_1 predictions (max |diff| in t/ha over folds)

- default_tabicl_cybench_DE: 0.1934
- default_tabicl_cybench_ES: 0.9939
- default_tabicl_cybench_FR: 0.1995
- default_tabicl_cybench_HU: 0.2749
- default_tabicl_cybench_IT: 0.7092
- default_tabicl_cybench_PL: 0.1649
- default_tabicl_cybench_RO: 0.09244
- scm_cybench_DE: 0.0004996
- scm_cybench_ES: 0.0004997
- scm_cybench_FR: 0.0004997
- scm_cybench_HU: 0.0004935
- scm_cybench_IT: 0.0004985
- scm_cybench_PL: 0.0004994
- scm_cybench_RO: 0.0004976
- wofost_cybench_DE: 0.000496
- wofost_cybench_ES: 0.0004952
- wofost_cybench_FR: 0.0004985
- wofost_cybench_HU: 0.0004962
- wofost_cybench_IT: 0.0004991
- wofost_cybench_PL: 0.0004975
- wofost_cybench_RO: 0.0004981
