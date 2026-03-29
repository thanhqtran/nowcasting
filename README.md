# midas_nowcasting
Nowcasting GDP with MIDAS.

Copied and fixed some bugs from [https://github.com/sapphire921/midas_pro](https://github.com/sapphire921/midas_pro).


This code forecasts Vietnam's GDP in 2026Q1, using PMI and/or PPI.


How to run:
- `main_1i.py` forecasts GDP in 2026Q1, based on PMI only.
- `main.py` forecasts based on both PMI and PPI.
- `main_ssm.py` constructs a state-space model of GDP and PMI, uses Kalman filter to estimate the relationship and then deliver forecasts.

Based on my experience, estimation based on PMI only yields more plausible results than including PPI.

The GDP dataset is seasonal adjusted with X13, using the IRIS Toolbox.