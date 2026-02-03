# SIESTA-LDP
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://github.com/judithspd/siesta-ldp/blob/main/LICENSE) 

This library implements three mechanims for ε-differential privacy and (ε, δ)-differential privacy. The mechanisms are implemented for being used under a local approach, adding noise directly to the raw data. 
Two types of mechanims are implemented: 
- For numerical records: _Laplace_ and _Gaussian mechanisms_. The implementation includes a final clipping applyied on the data with DP.
- For categorical records: _Exponential mechanism_.

This library provides dedicated function designed for being applied on both pandas dataframes and lists/numpy arrays. 

## Getting started
For applying DP mechanisms on your data you need to introduce:
* The **pandas dataframe** with the data.
* The **column** in the dataframe to be privatized.
* The **privacy budget (ε)**.
* The **probability of exceeding the privacy budget (δ)** in case of numerical attributes and the Gaussian mechanism.
* The **uper and lower bounds** for numerical attributes (optional).

**Example: apply DP to the [adult dataset](https://archive.ics.uci.edu/dataset/2/adult) with the Laplace mechanism for the column _age_ and the Exponential mechanism for the column _workclass_:**
```python
import pandas as pd
from ldp.numerical import dp_clip_laplace
from ldp.categorical import dp_exponential

# Read and process the data
data = pd.read_csv("examples/adult.csv")
data.columns = data.columns.str.strip()
cols = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "sex",
    "native-country",
]
for col in cols:
    data[col] = data[col].str.strip()

# Apply DP for the attribute age:
column_num = "age"
epsilon1 = 10
df = dp_clip_laplace(data, column_num, epsilon1, new_column=True)

# Apply DP for the attribute workclass:
column_cat = "workclass"
epsilon2 = 5
df = dp_exponential(data, column_cat, epsilon2, new_column=True)
```

### Warning
This project is under active development. 

## License
This project is licensed under the [Apache 2.0 license](https://github.com/judithspd/siesta-ldp/blob/main/LICENSE).

## Funding and acknowledgments
This work is funded by European Union through the SIESTA project (Horizon Europe) under Grant number [101131957](https://cordis.europa.eu/project/id/101131957).
<p>
<img align="center" width="250" src="https://raw.githubusercontent.com/SIESTA-eu/.github/main/profile/EN-Funded.jpg">
<img align="center" width="250" src="https://raw.githubusercontent.com/SIESTA-eu/.github/main/profile/logo.png">
<p>
