# YouTube Ad-Placement Advisor: A Data-Driven Decision Support Model

## Project Overview

This repository documents a course project in which YouTube video performance signals were transformed into a practical ad-placement decision support workflow. The work was structured to support business decision-making rather than pure “view prediction.” A proxy tier label was defined from inventory constraints (e.g., duration) and engagement heuristics, and a Random Forest classification model was trained to output tier recommendations together with probability estimates. A lightweight Streamlit interface was implemented to operationalize the model for user input and transparent interpretation.

The project artifacts include the dataset, analysis notebook, trained model files, generated figures, the Streamlit application, and the final academic report PDF.

## Dataset

Dataset name: YouTube analytics dataset (CSV)
Local file: youtube_analytics_dataset.csv
Unit of analysis: individual YouTube videos (n ≈ 537)

Core variables used (examples):
• view_count
• like_count
• comment_count
• duration_seconds (derived)
• engagement proxies (derived; e.g., likes/views, comments/views, engagement rate)

The dataset was used for exploratory analysis, feature engineering, tier proxy-labeling, and model training. The tier label should be interpreted as a decision-support proxy (heuristic), not as a causal measure of real ad outcomes.

## Data Preprocessing

The dataset was cleaned and transformed to improve interpretability and comparability across videos. The following preprocessing steps were applied:

• Parsing and normalization
– Video duration was converted into seconds (duration_seconds).
– Missing values were assessed and handled where relevant.

• Scale stabilization
– View count was log-transformed to reduce skew (e.g., log(1 + views)).

• Rate features (engagement proxies)
– likes_per_view and comments_per_view were computed.
– engagement_rate was computed as (likes + comments) / views.

All transformations were documented in the notebook, and the final feature set used for inference was saved (model_features.pkl) to ensure consistent ordering during deployment.

## Exploratory Data Analysis (EDA)

Exploratory analyses were conducted to characterize distributions, identify skew, and evaluate relationships between engagement signals and reach proxies.

Typical EDA outputs included:
• Distribution plots for key metrics (views, duration, engagement rates)
• Scatter plots illustrating relationships between views and engagement proxies
• Correlation heatmap summarizing linear associations among engineered features

Figures generated during EDA were exported and stored in the figures/ folder.

## Distribution Plots

Distribution plots were produced for:
• View count (raw and/or log scale)
• Video duration (seconds; bucketed for interpretability)
• Engagement-rate features (likes/views, comments/views, combined engagement)

These plots were used to motivate feature transformations (e.g., log scale) and to support the tiering logic based on inventory constraints.

## Scatter Plots (Relationship Analysis)

Scatter plots were used to explore monotonic or non-linear patterns between:
• Views and engagement proxies
• Duration and engagement rate
• Likes vs comments and their relationship to view count

These visuals supported the argument that raw reach metrics alone were insufficient for ad-placement decisions and that engagement proxies and inventory constraints should be considered jointly.

## Correlation Heatmap

A correlation heatmap was generated to summarize relationships among key engineered features. This analysis provided a compact diagnostic for feature redundancy and potential multicollinearity, and it motivated the selection of a robust tabular model that can tolerate correlated predictors.

## Machine Learning Model

Problem framing: classification for ad-placement tier recommendation (proxy-labeled)
Model family: Random Forest classifier

Rationale:
• Tabular performance and robustness to non-linear decision boundaries
• Reduced sensitivity to feature scaling compared with linear models
• Interpretable diagnostics via feature importance
• Availability of class probabilities to communicate prediction confidence

## Features used

The final feature set was composed of interpretable signals aligned with business decision logic. Typical features included:

• Duration / inventory constraint
– duration_seconds
– duration buckets (if used)

• Reach stabilization
– log_views

• Engagement proxies
– likes_per_view
– comments_per_view
– engagement_rate

The inference feature order was stored in: model_features.pkl

## Model Metrics

Model performance was evaluated using standard classification reporting procedures (e.g., accuracy and/or classification report). Where probabilities were used, the outputs were interpreted as a confidence indicator for operational decision-support rather than a guarantee of outcomes.

## Feature Importance (Random Forest)

Global feature importance was extracted from the trained Random Forest model to support interpretability. This analysis was used to communicate which signals most strongly influenced tier recommendations (e.g., engagement proxies versus duration constraints).

## Prediction Distribution (Actual vs Predicted)

A prediction distribution visualization was generated to compare proxy labels and model outputs and to check for systematic bias toward specific classes. Probability outputs were used to support transparent decision-making (e.g., high-confidence vs low-confidence recommendations).

## Streamlit Decision Support Application

A Streamlit app was implemented to allow users to input understandable metrics and receive:

• Predicted ad-placement tier
• Class probability distribution (confidence overview)
• Interpretability cues (e.g., feature importance reference)

Main application file: app.py
Trained model file: youtube_ad_tier_model.pkl
Feature mapping file: model_features.pkl

## Output Files

Key project outputs included:

• YouTube_Ad_Placement_Report.pdf
– Final academic report (methodology, results, limitations, references)

• youtube_video_project.ipynb
– End-to-end workflow (data loading → preprocessing → EDA → modeling)

• app.py
– Streamlit decision tool implementation

• youtube_ad_tier_model.pkl
– Trained Random Forest model artifact

• model_features.pkl
– Feature list/order for reliable inference

• figures/
– Exported plots used for analysis and reporting

## How to Run (Python)

1. Create and activate a virtual environment
   python -m venv .venv
   source .venv/bin/activate

2. Install dependencies
   pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib

3. Run the Streamlit application
   streamlit run app.py

## Academic Note on Scope and Interpretation

The tier label was defined as a proxy from operational  inventory and engagement. Therefore, results were interpreted as decision-support guidance rather than causal claims about campaign performance. The project emphasized transparency, interpretability, and reproducible workflow documentation.


## Author and Course Context

Author: Abhishek Verma
Program: M.Sc. Industrial Engineering & International Management (AI/ML)
Institution: Hochschule Fresenius University of Applied Sciences (Cologne Campus)
