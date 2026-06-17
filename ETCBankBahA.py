# app.py — Banking Service Preference Dashboard (FP-Growth)

import streamlit as st
import pandas as pd
import time

from mlxtend.frequent_patterns import (
    fpgrowth,
    association_rules
)

# --------------------
# PAGE CONFIG
# --------------------

st.set_page_config(
    page_title="Customer Banking Service Preference Analytics",
    layout="wide"
)

st.title(
    "Banking Service Preference Analysis Dashboard"
)

st.markdown(
"""
 Decision Support Dashboard
using FP-Growth Association Rule Mining
"""
)

# --------------------
# SIDEBAR
# --------------------

st.sidebar.header(
    "Configure Parameters"
)

support = st.sidebar.slider(
    "Minimum Support",
    0.10,
    1.00,
    0.30
)

confidence = st.sidebar.slider(
    "Minimum Confidence",
    0.10,
    1.00,
    0.60
)

lift = st.sidebar.slider(
    "Minimum Lift",
    1.00,
    5.00,
    1.20
)

max_len = st.sidebar.selectbox(
    "Maximum Itemset Length",
    [2,3,4],
    index=1
)

# --------------------
# FILE UPLOAD
# --------------------

uploaded = st.file_uploader(
    "Upload Encoded CSV Dataset",
    type=["csv"]
)

if uploaded:

    df = pd.read_csv(uploaded)

    st.success(
        "Dataset Uploaded"
    )

    c1,c2,c3 = st.columns(3)

    c1.metric(
        "Rows",
        df.shape[0]
    )

    c2.metric(
        "Columns",
        df.shape[1]
    )

    c3.metric(
        "Missing",
        df.isnull().sum().sum()
    )

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        df.head()
    )

    # --------------------
    # RUN MODEL
    # --------------------

    if st.button(
        "Run FP-Growth"
    ):

        start = time.perf_counter()

        frequent = fpgrowth(
            df,
            min_support=support,
            use_colnames=True,
            max_len=max_len
        )

        rules = association_rules(
            frequent,
            metric="confidence",
            min_threshold=confidence
        )

        rules = rules[
            rules["lift"] >= lift
        ]

        end = time.perf_counter()

        runtime = round(
            end-start,
            3
        )

        st.success(
            f"Completed in {runtime} sec"
        )

        st.subheader(
            "Frequent Itemsets"
        )

        st.dataframe(
            frequent
        )

        st.subheader(
            "Association Rules"
        )

        if len(rules)>0:

            display = rules[
                [
                    "antecedents",
                    "consequents",
                    "support",
                    "confidence",
                    "lift"
                ]
            ]

            st.dataframe(
                display
            )

            st.subheader(
                "Rule Ranking"
            )

            st.bar_chart(
                display["lift"]
            )

            csv = (
                display
                .to_csv(index=False)
                .encode()
            )

            st.download_button(
                "Download Rules",
                csv,
                "fp_rules.csv"
            )

        else:

            st.warning(
                "No rules generated."
            )

