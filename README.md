# MohanLab Project Summary (May 2024 - Present) 

> **Important Note:** This repository contains selected portions of the project, including the application of *Lasso Regression* for biomarker optimization, along with data visualization conducted using tools like BayesiaLab and Google Sheets.

## Identification of **Key** Biomarkers in Patients with Lupus Nephritis

## Overview
This project involves advanced data analysis and machine learning techniques applied to biomedical research, with a specific focus on identifying biomarkers for Lupus Nephritis. Our objective is to support ongoing efforts in the quest to find effective treatments and potentially a cure for this kidney disease.

## Objectives
- **Lasso Regression for Biomarker Optimization:**  
   Applied Lasso Regression to improve the accuracy of biomarker data, shifting from traditional models to a binary classification approach to optimize biomarker panels for the highest possible ROC AUC scores.
  
- **Visualization of Protein Relationships in Lupus Nephritis:**  
   Leveraging BayesiaLab, I analyzed and visualized complex relationships between various proteins within Lupus Nephritis data. This work enhances the understanding of target genes associated with Lupus and refines biomarker panels.

- **Data Aggregation Across Diverse Databases:**  
   Aggregated and synthesized plots from over ten different databases, providing a broad perspective on the impacts of Lupus across varied patient demographics. This compilation aids researchers by revealing patterns that could inform more effective treatments.

## Project Highlights
This project has resulted in a comprehensive set of resources and insights, including:
- **Enhanced Biomarker Panels** through data-driven selection methods, emphasizing proteins and genetic markers relevant to Lupus Nephritis.
- **Detailed ROC Curves** for specific protein comparisons, including CRC (Colorectal Cancer) vs. HC (Healthy Controls), to assess diagnostic accuracy.
- **Cross-Platform Data Synthesis:** Finalized a spreadsheet of aggregated plots, facilitating a clear view of trends for potential reference in multiple forthcoming publications.

## Model Architecture and Tools
The methodology integrates multiple tools and approaches:
- **Lasso Regression:** Optimizes biomarker selection by emphasizing features that contribute significantly to ROC AUC scores.
- **BayesiaLab:** Used for visualizing relationships between proteins, aiding in the understanding of gene-protein interactions within Lupus Nephritis.
- **Data Visualization in Google Sheets:** Synthesized complex data across patient demographics for a comprehensive visual analysis.

![Trial_Lasso_Binary_Classification_Top_N](https://github.com/user-attachments/assets/d7e24399-023b-4919-b10f-c146352ab880)

## References

[1] Robert Tibshirani. *Regression shrinkage and selection via the lasso.* Journal of the Royal Statistical Society, 58(1):267–288, 1996.

## Requirements
- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Keras
- Matplotlib (for plotting ROC curves)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/UniversityOfHouston_MohanLab_DataScienceInternship.git
   cd UniversityOfHouston_MohanLab_DataScienceInternship
