# MultiNano Pretrained Models Guide

You can download all MultiNano pretrained models from the following link:  
[**MultiNano Models Drive**](https://drive.google.com/drive/folders/18vjG8KKiuw8K0IBtQWP_CQ55vHZOTbFF?usp=drive_link)

## Model Categories and Recommendations

### 1. Site-level Models

We provide two site-level models:

- **HEK293T-trained model** (recommended for *human* and *animal* nanopore datasets)
- **Arabidopsis-trained model** (recommended for *plant* nanopore datasets)

> **Recommendation**:  
Use the **HEK293T-trained model** for mammalian or human data, and the **Arabidopsis-trained model** for plant data. These models are fine-tuned for their respective species and offer better prediction accuracy.

---

### 2. Read-level Models

We provide read-level models trained on the following datasets:

- **IVT dataset**
- **Synthetic RNA dataset**

These models are suitable for different scenarios and help capture the signal features at the single-read resolution.

---

### 3. False Positive Control Models

We offer **two models** specifically designed to control the false positive rate:

- Both models are effective in minimizing false positives.
- For **higher confidence in methylation site identification**, we recommend **predicting with both models and taking the intersection of their results**.

> **Tip**:  
Using both models in combination does not significantly increase runtime, but it can greatly improve the accuracy of your predicted methylation sites.

---
