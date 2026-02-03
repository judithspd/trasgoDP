# -*- coding: utf-8 -*-

# Copyright 2026 Spanish National Research Council (CSIC)
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

"""Laplace mechanism for local DP."""

import numpy as np
import pandas as pd

def dp_clip_laplace(
    df: pd.DataFrame,
    column: str,
    epsilon: float,
    lower_bound=None,
    upper_bound=None,
    new_column=False,
) -> pd.DataFrame:
    "Apply the Laplace mechanims to a numeric column of a dataframe"

    if column not in df.keys():
        raise ValueError("Column: {column} not in the dataframe.")

    if epsilon <= 0:
        raise ValueError("The privacy budget must be greater than 0.")

    if np.issubdtype(df[column].dtype, np.integer):
        data = df[column].astype(int)
    elif np.issubdtype(df[column].dtype, np.floating):
        data = df[column].astype(float)
    else: 
        raise ValueError("Type of the column not allowed for the Laplace mechanism.")

    if lower_bound is None:
        lower_bound = min(data.values)
    if upper_bound is None:
        upper_bound = max(data.values)

    clipped = data.clip(lower_bound, upper_bound)

    sensitivity = upper_bound - lower_bound
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, size=len(clipped))

    dp_column = clipped + noise

    if np.issubdtype(df[column].dtype, np.integer):
        dp_column = round(dp_column, 0).astype(int)

    dp_column = np.clip(dp_column, lower_bound, upper_bound)
    if new_column:
        df[f"dp_{column}"] = dp_column
    else:
        df[column] = dp_column

    return df
