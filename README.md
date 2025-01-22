# Biodiversity Analysis and Conservation Insights

## Overview

This project focuses on analyzing biodiversity data to derive actionable insights for conservation efforts. It integrates and evaluates species observations and taxonomic data to identify patterns, trends, and relationships in biodiversity metrics across various national parks.

---

## Datasets

1. **Observations Dataset**:
   - Contains records of species observations in various national parks.
   - Key fields: scientific names, park names, observation counts.

2. **Species Info Dataset**:
   - Provides taxonomic classifications and conservation statuses.
   - Key fields: taxonomic categories, scientific names, common names, and conservation statuses.

---

## Objectives

- To analyze biodiversity data and uncover patterns across parks.
- To study the distribution of species and their conservation statuses.
- To identify associations between species categories and conservation statuses.

---

## Key Findings

1. **Dataset Insights**:
   - Total species records: 5,824.
   - Conservation status available for only 191 species.
   - Categories include Mammals, Birds, Reptiles, and more.

2. **Statistical Analysis**:
   - Chi-square test identified significant associations between species categories and conservation statuses (p-value ≈ 0.026).
   - Mammals and vascular plants are key drivers of this association.

3. **Park Diversity**:
   - Yellowstone National Park has the most even species distribution.
   - Great Smoky Mountains exhibits slight imbalances in species distribution.

---

## Methods

- **Data Cleaning**:
  - Handled missing values, duplicates, and ensured data consistency.
- **Exploratory Data Analysis (EDA)**:
  - Visualized trends, identified patterns.
- **Dataset Integration**:
  - Combined observation data with taxonomic and conservation status information.
- **Statistical Analysis**:
  - Chi-square and post-hoc tests to assess category-conservation status relationships.

---

## How to Use

1. **Code**:
   - The analysis is provided in the `biodiversity.ipynb` notebook.
   - Ensure Python libraries such as Pandas, NumPy, Matplotlib, and Scipy are installed.
   - Run `pip install -r requirements.txt` to install necessary dependencies.
2. **Data**:
   - Place `observations.csv` and `species_info.csv` in the working directory.
   - Update file paths in the notebook if necessary.
3. **Presentation**:
   - A summary of the findings is available in `biodiversity_presentation.pptx`.

---

## Next Steps
- Expand conservation status data coverage to include more species.
- Conduct regional-level analysis for targeted conservation strategies.
- Develop predictive models for species at risk based on observed trends.
