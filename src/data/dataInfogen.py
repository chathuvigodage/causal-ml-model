import pandas as pd
import os

# Define folder to save CSV
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# Define the data
columns = ["Variable_Name", "Variable_Type", "Desc"]
rows = [
    ["InterestRate", "dosage", "Treatment (interest rate)"],
    ["LoanApproved", "target", "Loan approval outcome"],
    ["LoanAmount", "float", "Loan characteristic"],
    ["LoanDuration", "float", "Loan characteristic"],
    ["TotalDebtToIncomeRatio", "float", "Loan characteristic"],  # ✅ FIXED
    ["CreditScore", "float", "Loan characteristic"],
    ["NumberOfOpenCreditLines", "int", "Loan characteristic"],
    ["AnnualIncome", "float", "Financial background"],
    ["SavingsAccountBalance", "float", "Financial background"],
    ["TotalLiabilities", "float", "Financial background"],
    ["Age", "int", "Socioeconomic background"],
    ["EducationLevel", "cat", "Socioeconomic background"],
    ["MaritalStatus", "cat", "Socioeconomic background"],
    ["EmploymentStatus", "cat", "Socioeconomic background"],
    ["PaymentHistory", "float", "Bank relationship"]
]

# Create DataFrame
df = pd.DataFrame(rows, columns=columns)

# Save to CSV (IMPORTANT: correct filename)
csv_path = os.path.join(data_dir, "loan_data_info.csv")
df.to_csv(csv_path, index=False)

print(f"CSV generated at: {csv_path}")
