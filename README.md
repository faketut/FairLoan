# FairLoan: Auditing AI for Bias in Mortgage Decisions

Welcome to **FairLoan** – a project for the AI Bias Bounty Hackathon.

## 🚀 Project Summary

FairLoan predicts mortgage loan approvals and rigorously audits the model for bias across gender, race, income, age, and location. Our goal is to build not just an accurate model, but a fair one—ensuring that automated decisions do not perpetuate discrimination or social inequality.

## 🏆 Key Features
- **Data Cleaning & Feature Engineering:** Handles missing values, encodes categorical variables, and creates features like income-to-loan ratio.
- **Balanced Modeling:** Uses Random Forest with SMOTE to address class imbalance.
- **Bias & Fairness Auditing:** Employs Fairlearn to measure demographic parity, selection rates, and group disparities.
- **Visual Evidence:** Generates clear charts for approval rates and fairness metrics by group.

## 📊 Results Snapshot
- **Validation F1 Score:** 0.66
- **Key Bias Findings:**
  - Males approved at 36%, females at 24%
  - Black and Native American applicants have the lowest approval rates (16–18%)
  - Strong bias toward higher-income applicants (demographic parity difference for income = 1.0)
  - Residents of historically redlined areas are less likely to be approved

## 📂 Project Structure
| File/Folder         | Purpose                                                      |
|---------------------|--------------------------------------------------------------|
| loan_model.py       | Main Python script for data processing, modeling, and bias analysis |
| loan_model.ipynb    | Jupyter notebook for EDA, modeling, and reporting            |
| requirements.txt    | Python dependencies for the project                          |
| outputs/            | Folder for generated outputs (e.g., submission.csv, reports) |
| charts/             | Folder for generated charts and visualizations               |
| datasets/           | Provided datasets (train/test)                               |
| resources/          | Templates and reference materials                            |

## 🛠️ Getting Started

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the workflow:**
   - For a step-by-step approach, use `loan_model.ipynb` (Jupyter Notebook)
   - For a full pipeline, run:
     ```bash
     python loan_model.py
     ```
3. **Review outputs:**
   - Predictions: `outputs/submission.csv`
   - Fairness charts: `charts/`
   - Technical report: `ai_risk_report.md` or `.docx`

## 📑 Submission Checklist
- [x] Source code (`loan_model.py`, `loan_model.ipynb`)
- [x] Output file (`outputs/submission.csv`)
- [x] Technical report (`ai_risk_report.md` or `.docx`)
- [x] Visual evidence (`charts/`)
- [x] README (this file)

## 💡 About the Challenge
This project was built for the [AI Bias Bounty Hackathon](https://ai-bias-bounty-hackathon.devpost.com/), where the goal is to detect, explain, and mitigate bias in AI models using real-world data.

## 👤 Author
Jian Feng

---

*Let's build AI that's not just smart, but also fair.*
