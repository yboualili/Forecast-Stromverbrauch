# BDA - Heat Pump Data Analysis

## TL;DR
Predict cumulative `S_TOT` for heatpumps for all single-family houses.

## Data Source
Raw data can be found here: https://zenodo.org/record/5642902#.YiDd6pPMKWB

A paper explaining the data set can be found here:https://www.nature.com/articles/s41597-022-01156-1

Reference code can be found here: https://github.com/ISFH/WPuQ

## Relevant Shorthands

- SFH: single-family house

- S: apparent power, the actual power drawn from the grid and paid for (measured in VA bzw. W) [Prediction Goal]
- Q: reactive power, the "useless foam" ([see this analogy](https://youtu.be/Tv_7XWf96gg?t=53)) (measured in VAr)
- P: active power, the work is actually done by the consumer, e.g. heating (measured in W)
- PF: active power / apparent power, "value for payment"
- U: voltage, measured in Volts
- I: current, measured in Amps

[Source](https://www.nature.com/articles/s41597-022-01156-1/tables/5)

### Example: Structure of Household 10

```
SFH10
  |- HEATPUMP
    |- table
      |- index
      |- S_TOT
      |- PF_TOT
      |- P_1
      |- P_2
      |- P_3
      |- P_TOT
      |- Q_1
      |- Q_2
      |- Q_3
      |- Q_TOT
  |- HOUSEHOLD
```

### Heatpump vs Household

![Measurement of Data](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41597-022-01156-1/MediaObjects/41597_2022_1156_Fig2_HTML.png?as=webp)

Heatpumps and households are independently measured.
See the image above.
As heatpumps are new consumers, we are probably interested in their load profiles in the first place.
They are novel in a way that loads cannot be immediately predicted.
Thus, we want to predict the cumulative **apparent power** (`S_TOT`, see above) of households.
In a later step, we can add predictions of households.
But we must assume that electricity providers are already pretty familiar with standard load profiles (they have seen them for years).

## Training a Model: Shape of In- and Output (For the NN approach done by Jasper. Könnt ihr erst mal ignorieren)

Data exists in a resolution of 15min (4/1h, 96/1day).


**Possible Inputs are:**
1. Cumulative `S_TOT` for all SFH in 15min resolution for one day. For three days in a row. I.e. (96,2,3) | (96 x [`S_TOT`, Time], 3 Days)
2. Cumulative `S_TOT` for all SFH in 15min resolution for three days. (288,2,1) | (288 x (`S_TOT`, Time))
2. `S_TOT` for all SFH in 15min resolution for three days. (288,2,No.SFH) | (288 x (`S_TOT`, Time), No. SFH)

**Prediction Goal:**

`S_TOT` in 15min resolution for one day; i.e. (96,1,1)
