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

3. **Before/After Comparison**
   - Compares raw and filtered datasets.
   - Reports tweet counts, unique accounts, follower statistics, engagement metrics, keyword distribution, region distribution, and filter decisions.

4. **Organized CSV Export**
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

- `.gitignore`  
  Excludes `.env`, generated CSV files, and local output folders.

## Methodological Note

The rule-based verification layer is used to keep the prototype transparent and explainable. Media accounts, political actors, and government institutions are treated as contextual sources rather than direct citizen aspirations. Education-focused fess communities are retained with lower weight because they represent student or applicant discussion spaces, not individual respondents.

## Disclaimer

This is a prototype dataset pipeline for essay and demonstration purposes. Any policy conclusion should be supported with larger-scale sampling, multi-platform data, official complaint channels, and offline validation.
