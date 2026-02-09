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

"""Utility loss metric and divergence between distributions."""

import numpy as np
import pandas as pd
import typing
import copy

import scipy


def correlation_loss(
    df_original: pd.DataFrame,
    df_dp: pd.DataFrame,
    features: typing.Optional[typing.List[str]] = None,
    method: str = "pearson",
    new_column: bool = False,
) -> float:
    """Compute utility loss (%) based on the preservation of the correlation
    between features.

    :param df_original: dataframe with the original data.
    :type df_original: pandas dataframe

    :param df_original: dataframe with the data privatized using DP.
    :type df_original: pandas dataframe

    :param features: list of featured for calculating the correlation.
    :type features: list

    :param method: method for calculating the correlation.
    :type method: string

    :param new_column: boolean, default to False. If True, the columns with dp
    start with dp_. Otherwise, the names of the columns with DP are the same as in
    the original dataset.
    :type  new_column: boolean

    :return: utlity loss (%) comparing the difference between correlations.
    :rtype: float
    """
    if not all(col in df_original.columns for col in features):
        raise ValueError("Not all features are in the dataframe.")
    if not all(col in df_dp.columns for col in features):
        raise ValueError("Not all features are in the dataframe.")

    if method not in ["pearson", "kendall", "spearman"]:
        raise ValueError("Method not allowed for calculating the correlation.")

    if new_column:
        new_features = []
        for col in features:
            dp_col = f"dp_{col}"
            if dp_col in features:
                new_features.append(dp_col)
            else:
                new_features.append(col)
    else:
        new_features = features

    categorical_cols = df_original.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        unique_vals = pd.concat([df_original[col], df_dp[col]]).unique()
        mapping = {val: i for i, val in enumerate(unique_vals)}
        df_original[col] = df_original[col].map(mapping)
        if new_column:
            dp_col = f"dp_{col}"
            df_dp[dp_col] = df_dp[dp_col].map(mapping)
        else:
            df_dp[col] = df_dp[col].map(mapping)

    X_original = df_original[features]
    X_dp = df_dp[new_features]

    corr_original = X_original.corr(method=method).values
    mask = ~np.eye(corr_original.shape[0], dtype=bool)
    corr_original = corr_original[mask]
    corr_dp = X_dp.corr(method=method).values
    mask = ~np.eye(corr_dp.shape[0], dtype=bool)
    corr_dp = corr_dp[mask]

    diff = np.abs(corr_original - corr_dp)

    return 100 * np.mean(diff) / np.mean(np.abs(corr_original))


def divergence_distributions(
    df_original: pd.DataFrame,
    df_dp: pd.DataFrame,
    column: str,
    new_column: bool = False,
) -> dict:
    """Compute divergence between the distribution of a feature in the original
    and DP datasets.

    :param df_original: dataframe with the original data.
    :type df_original: pandas dataframe

    :param df_original: dataframe with the data privatized using DP.
    :type df_original: pandas dataframe

    :param column: column to which DP has been applied.
    :type column: string

    :param new_column: boolean, default to False. If True, the column with dp
    starts with dp_.
    :type  new_column: boolean

    :return: dictionary with the divergence metrics (TVD, JS, KL).
    :rtype: dict
    """
    if column not in df_original.keys():
        raise ValueError("Column: {column} not in the original dataframe.")
    if column not in df_dp.keys():
        raise ValueError("Column: {column} not in the dataframe with DP.")

    if new_column:
        dp_column = f"dp_{column}"
    else:
        dp_column = column

    freq_orig = df_original[column].value_counts().sort_index()
    freq_dp = df_dp[dp_column].value_counts().sort_index()
    all_categories = sorted(set(freq_orig.index).union(set(freq_dp.index)))
    freq_orig = freq_orig.reindex(all_categories, fill_value=0)
    freq_dp = freq_dp.reindex(all_categories, fill_value=0)

    P = freq_orig / freq_orig.sum()
    Q = freq_dp / freq_dp.sum()

    tvd = 0.5 * np.sum(np.abs(P - Q))
    js = scipy.spatial.distance.jensenshannon(P, Q, base=np.e) ** 2
    kl = scipy.stats.entropy(P, Q)

    return {"tvd": tvd, "js": js, "kl": kl}
