# High-Performance Concrete Mix Design Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code for the paper:  
**"Automating High-Performance Concrete Mix Design: A Deep Learning Framework for Inverse Prediction of Composition from Target Strength and Age"**

## Overview

We implement five machine learning models for inverse prediction of HPC mix proportions from target strength and age:

- **Deep Learning** (multi-tower MLP architecture with cross-connected towers)
- **Support Vector Regression (SVR)**
- **XGBoost**
- **Random Forest**
- **Multi-output Linear Regression**

## Dataset

The [Concrete Compressive Strength dataset](https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength) from the UCI Machine Learning Repository (Yeh, 1998) is used:
- 1,030 observations
- 8 input features → 7 output components (cement, slag, fly ash, water, admixture, coarse aggregate, fine aggregate)

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
