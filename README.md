# midas_nowcasting
Nowcasting GDP with MIDAS.

Copied and fixed some bugs from [https://github.com/sapphire921/midas_pro](https://github.com/sapphire921/midas_pro).


This code forecasts Vietnam's GDP in 2026Q1, using PMI and/or PPI.


How to run:
- `main_1i.py` forecasts GDP in 2026Q1, based on PMI only.
- `main.py` forecasts based on both PMI and PPI.

Based on my experience, estimation based on PMI only yields more plausible results.

The GDP dataset is seasonal adjusted with x13, using the IRIS Toolbox.