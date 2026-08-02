# Algorithmic Fairness & Bias Mitigation in AI Recruitment

This repository contains a controlled Python simulation designed to stress-test machine learning models applied to Human Resources (People Analytics) and recruitment tasks. It explores how historical biases against specific groups are replicated by standard ML models and evaluates the effectiveness of mathematical mitigation techniques like **Sample Reweighing**.

## 📌 Project Overview
When machine learning models are trained on historical hiring data, they risk codifying and amplifying past human biases. This project simulates a hiring pipeline where:
1. **Historical Bias** is explicitly injected into past hiring decisions (penalizing candidates based on gender and career gaps).
2. A **Gender-Blind Model** is deployed to see if simply removing the demographic feature solves the bias (spoiler: it doesn't).
3. A **Reweighted Gender-Blind Model** is implemented using conditional probability adjustments to mitigate the bias before training.
4. A **Merit Benchmark** acts as the baseline for "true potential" to measure how much utility/accuracy we lose or gain.

---

## 🛠️ Simulation Specifications & Methodology
* **Sample Size ($N$):** 1,500 candidates per run.
* **Simulation Runs:** 30 independent iterations (to ensure statistical robustness and calculate variance).
* **Selection Rate:** Top 30% of candidates are invited for interviews based on predicted probabilities.
* **Mitigation Technique:** Disparate Impact mitigation via **Sample Reweighing** (adjusting sample weights based on the joint probability of demographic groups and historical labels).

---

## 📊 Evaluation Metrics
The models are evaluated across 30 runs on two main vectors: **Predictive Performance** and **Fairness/Equity Metrics**:
* **Merit Accuracy & Recall:** How well the model identifies truly qualified individuals.
* **Selection Rate Ratio:** The interview selection rate of Group 0 (Female) divided by Group 1 (Male) — targeting the 80% rule ($0.80$).
* **Equal Opportunity Gap:** The difference in True Positive Rates (TPR) between groups.
