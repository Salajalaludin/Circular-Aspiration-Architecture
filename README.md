# Circular Aspiration Architecture

Prototype data pipeline for collecting, verifying, and preparing digital public aspirations on education access in Jakarta and Banten.

## Overview

This repository contains a proof-of-concept implementation of the **Circular Aspiration Architecture**, a data-driven framework for monitoring public education issues from social media. The prototype focuses on Jakarta and Banten, with emphasis on education access, school admission, scholarships, tuition, and higher education participation.

The project is designed for essay/prototype purposes. It does not claim to represent the entire population; instead, it demonstrates how digital public conversations can be collected, verified, filtered, and prepared for further NLP analysis.

## Pipeline

1. **Scraping**
   - Collects public X/Twitter posts using Apify.
   - Uses education-related keywords for Jakarta and Banten.
   - Stores region and keyword metadata for each tweet.

2. **Data Verification Layer**
   - Applies transparent rule-based classification.
   - Separates citizen candidates, education fess communities, general fess accounts, media, government institutions, political actors, and irrelevant sources.
   - Produces analysis-ready data after filtering contextual sources and irrelevant noise.
   - Requires education evidence in tweet text that aligns with the retrieval query; the query itself is not treated as relevance evidence.

3. **Before/After Comparison**
   - Compares raw and filtered datasets.
   - Reports tweet counts, unique accounts, follower statistics, engagement metrics, keyword distribution, region distribution, and filter decisions.

4. **Preprocessing and IndoBERT Sentiment Analysis**
   - Preserves sentiment-bearing hashtags, negation, and mapped emoji signals.
   - Audits spam, short text, exact duplicates, and uncertain predictions.
   - Stores all class probabilities and computes correctly normalized weighted scores.
   - Generates a blind manual-validation sample for confusion matrix and macro-F1 evaluation.

5. **LDA Topic Modeling**
   - Removes near-duplicates and builds region-balanced pseudo-documents from accepted non-neutral discourse while inferring every eligible tweet.
   - Selects topic count using NPMI coherence, seed stability, topic diversity, and held-out perplexity.
   - Reports soft topic prevalence weighted by source type across region, keyword, and sentiment.
   - Exports representative tweets and a manual topic-naming template.

6. **Representation Weighting and Coverage Diagnostics**
   - Reduces observed domination by prolific accounts and uneven query yields without inferring user demographics.
   - Compares unweighted, source-only, operational, and one-account sensitivity estimates.
   - Reports weight trimming, effective sample size, account contribution, and account-clustered bootstrap intervals.
   - Only enables population calibration when a complete, sourced, and explicitly verified regional reference file is supplied.

7. **Organized CSV Export**
   - The verification notebook can export cleaner CSV tables into grouped folders:
     - raw core tweets,
     - verified labels,
     - analysis-ready tweets,
     - comparison summaries.

## Files

- `01_scraping_twitter.ipynb`
  Scrapes X/Twitter data using Apify and saves raw tweet data.

- `02_data_verification_rule_based.ipynb`
  Applies rule-based verification, relevance filtering, before/after comparison, and organized CSV export.

- `03_preprocessing_sentiment.ipynb`
  Preprocesses verified tweets, runs three-class IndoBERT sentiment inference, audits uncertainty, and exports weighted summaries.

- `04_lda_topic_modeling.ipynb`
  Discovers latent issues with LDA, evaluates topic quality, and exports weighted topic distributions and audit tables.

- `05_demographic_weighting.ipynb`
  Performs representation weighting, coverage diagnostics, sensitivity analysis, and account-clustered uncertainty estimation. Despite the legacy filename, it does not infer individual demographics.

- `06_dashboard.py`
  Provides an interactive Streamlit dashboard for weighted sentiment, topic prevalence, query coverage, uncertainty, and tweet exploration.

- `requirements.txt`
  Lists the Python dependencies required by the notebooks.

- `.gitignore`
  Excludes `.env`, generated CSV files, and local output folders.

## Setup

Install the notebook dependencies in the same Python environment used by the Jupyter kernel:

```bash
python -m pip install -r requirements.txt
```

Run the dashboard after notebooks 03-05 have produced their organized outputs:

```bash
streamlit run 06_dashboard.py
```

## Methodological Note

The rule-based verification layer is used to keep the prototype transparent and explainable. Media accounts, political actors, and government institutions are treated as contextual sources rather than direct citizen aspirations. Education-focused fess communities are retained with lower weight because they represent student or applicant discussion spaces, not individual respondents.

The weighting stage corrects only observable imbalances within the collected sample. Query-region metadata indicates the retrieval stratum, not the author's residence. Therefore, weighted estimates must not be presented as population opinions or as proof of demographic representativeness.

## Disclaimer

This is a prototype dataset pipeline for essay and demonstration purposes. Any policy conclusion should be supported with larger-scale sampling, multi-platform data, official complaint channels, and offline validation.
