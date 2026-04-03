# Benchmarks

Galpy Orbit Integration Performance on a AMD Ryzen 7 7700X 8-Core CPU.


## Orbit Integration Benchmark
### Single timestep

0 and t_end from t_end = 10-9 to 1.

30 samples per each set of parameters.


![Orbit Benchmark Plot](./orbit_calculation_time_vs_time_step.png)
***Figure A***

| Time Step (t_end) | Avg Time (1M stars, sec) | Avg Time (2M stars, sec) |
| --- | --- | --- |
| 1e-09 | 0.700 | 1.380 |
| 1e-08 | 0.703 | 1.426 |
| 1e-07 | 0.712 | 1.413 |
| 1e-06 | 0.715 | 1.405 |
| 1e-05 | 0.706 | 1.389 |
| 1e-04 | 0.701 | 1.391 |
| 0.001 | 0.701 | 1.408 |
| 0.01 | 0.700 | 1.403 |
| 0.1 | 0.701 | 1.424 |
| 1.0 | 0.702 | 1.392 |



### 2 Million stars vs 2 Thousand stars

5 timesteps.

Time range: 0 to t_end * 5

Ex. t_end = 0.1, 5 timesteps generated from 0 to 0.5


![2M vs 2k 5 Timesteps Benchmark](./orbit_5by5.png)
***Figure B***

| t_end | Average time_taken (200k) | Average time_taken (2M) |
| --- | --- | --- |
| 1e-09 | 0.3849 | 4.1957 |
| 1e-08 | 0.3856 | 4.1963 |
| 1e-07 | 0.3859 | 4.1471 |
| 1e-06 | 0.3865 | 4.0515 |
| 1e-05 | 0.3857 | 4.0439 |
| 1e-04 | 0.3865 | 4.0365 |
| 0.001 | 0.3920 | 4.1096 |
| 0.01 | 0.4067 | 4.2275 |
| 0.1 | 1.1458 | 11.6065 |
| 1.0 | 6.8232 | 68.4154 |


### 2 Million Stars, Compare Shifts and Step 1,2,4

![Steps/Shifts Comparison](./Steps%20&%20Shifts%20Comparisons.png)
***Figure C***

### Markdown Table

|       t_end | Average time_taken (2Million, 1 step) | Average time_taken (2Million, 1 step, t shift + 2) | Average time_taken (2Million, 1 step, t_shift -2) | Average time_taken (2Million, 2 steps) | Average time_taken (2Million, 4 steps) | Average time_taken (2Million, 8 steps) | Average time_taken (2Million, 16 steps) | Average time_taken (2Million, 32 steps) |
| ----------: | ------------------------------------: | -------------------------------------------------: | ------------------------------------------------: | -------------------------------------: | -------------------------------------: | -------------------------------------: | --------------------------------------: | --------------------------------------: |
| 0.000000001 |                                1.3834 |                                             1.3938 |                                       1.403595591 |                            3.255156422 |                            3.842557001 |                            4.713445807 |                             6.498007393 |                             9.868268919 |
|  0.00000001 |                                 1.369 |                                             1.3788 |                                       1.406479502 |                            3.254553604 |                            3.912492561 |                             4.83469758 |                             6.520039034 |                             9.405585575 |
|   0.0000001 |                                1.3744 |                                             1.3812 |                                       1.382572079 |                            3.275227594 |                            3.974252415 |                            4.737000608 |                             6.514193201 |                             9.396988297 |
|    0.000001 |                                1.3749 |                                             1.3819 |                                       1.406913471 |                             3.27682786 |                             3.92817626 |                            4.752181959 |                             6.295022249 |                             9.365447235 |
|     0.00001 |                                1.3713 |                                             1.3802 |                                        1.43025713 |                            3.246943235 |                            3.932993412 |                            4.785920143 |                             6.248439503 |                             9.393179274 |
|      0.0001 |                                1.3749 |                                             1.3826 |                                       1.383241367 |                            3.267659855 |                             3.92597537 |                            4.866756439 |                              6.25447011 |                              9.42825098 |
|       0.001 |                                1.3753 |                                             1.3828 |                                       1.401276207 |                            3.264143133 |                            4.059873199 |                            4.877811909 |                             6.247785616 |                             9.431434059 |
|        0.01 |                                1.3728 |                                             1.3958 |                                        1.39389863 |                            3.303942537 |                            4.063896704 |                             4.77941227 |                             6.326437521 |                             9.500870466 |
|         0.1 |                                1.3757 |                                             1.4073 |                                       1.385934114 |                            4.159876394 |                             4.87308917 |                            5.668350887 |                             7.172400284 |                              10.3154161 |
|           1 |                                1.4642 |                                             1.4544 |                                        1.38934803 |                            18.85924973 |                            19.56199803 |                            20.32872915 |                             21.41473475 |                             24.57249374 |


### Conclusions

- Won't be trivial to add motion to the visualizer while maintaining 30-60 fps.
- Very notable time cost increase from 0.01 to 0.1 Gyr.
- No gain from partitioning the data from 2 Million stars to sets of X stars
- Shifting time along the time line doesn't affect integration cost.
    - No point in using data closer to the integration target time range
- Significant jump from 1 timestep to 2 timestep integration
    - Might be worth integrating 1 timestep at a time
- Time range (*abs(t_start - t_end)*) increases also significantly increases time taken (Figure B vs Figure C data)


