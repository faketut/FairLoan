
Hello, judges and fellow participants!
I am Jian Feng, and my project is “Fair Loan Approval: Auditing AI for Bias in Mortgage Decisions.”
Loan approval algorithms shape real lives—determining who gets access to financial opportunities.
But if these models are biased, they can reinforce social inequalities and exclude vulnerable groups.
my challenge: Build a predictive model for loan approval, and rigorously audit it for bias across gender, race, income, age, and location.
I started by exploring the dataset, identifying protected attributes like Gender, Race, Income, Age, and Zip Code Group.
I cleaned the data, engineered features such as the income-to-loan ratio, and balanced the classes using SMOTE to ensure fairer training.
For modeling, I chose a Random Forest classifier, tuned for both accuracy and fairness.
I evaluated the model using standard metrics—accuracy, precision, recall, and F1 score—achieving an F1 of 0.66 on validation data.
But I didn’t stop at performance.
I used the Fairlearn library to audit my model’s predictions for group fairness, calculating metrics like demographic parity difference and selection rates for each group.

my analysis revealed significant disparities:
Gender: Males Ire approved at a rate of 36%, while females Ire approved at just 24%.
Race: Black and Native American applicants had the loIst approval rates, around 16–18%, compared to 36% for White applicants.
Income: Approval rates varied dramatically by income, with a demographic parity difference of 1.0—indicating strong bias toward higher-income applicants.
Location: Applicants from historically redlined areas Ire less likely to be approved.
I visualized these disparities with clear bar charts, making the bias easy to see and understand.

If deployed as-is, this model would risk perpetuating discrimination—denying loans to already disadvantaged groups.
This highlights the importance of not just building accurate models, but also auditing and mitigating bias before deployment.

In the future, I’d explore advanced fairness mitigation and explainability tools to further improve equity.
Thank you for ymy attention—let’s build AI that’s not just smart, but also fair!