import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Set seed for reproducibility
np.random.seed(42)

# Generate a synthetic dataset of companies
num_samples = 1000
industries = ["FinTech", "HealthTech", "AI", "E-commerce", "EdTech", "AgriTech", "Gaming", "Cybersecurity"]
locations = ["US", "UK", "India", "Germany", "France", "Canada", "China", "Singapore"]

# Randomly generate features
data = {
    "Company": [f"Company_{i}" for i in range(num_samples)],
    "Industry": np.random.choice(industries, num_samples),
    "FundingAmountUSD": np.random.normal(50_000_000, 20_000_000, num_samples).astype(int),
    "EmployeeCount": np.random.randint(10, 5000, num_samples),
    "Age": np.random.randint(1, 30, num_samples),
    "Location": np.random.choice(locations, num_samples),
    "Acquired": np.random.choice([0, 1], num_samples, p=[0.7, 0.3])  # Imbalanced: 70% not acquired
}

df = pd.DataFrame(data)

# Split into training and testing datasets
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["Acquired"], random_state=42)
test_df1, temp_df2 = train_test_split(temp_df, test_size=0.66, stratify=temp_df["Acquired"], random_state=43)
test_df2, test_df3 = train_test_split(temp_df2, test_size=0.5, stratify=temp_df2["Acquired"], random_state=44)


test_features1 = test_df1.drop(columns=["Acquired"])
test_features2 = test_df2.drop(columns=["Acquired"])
test_features3 = test_df3.drop(columns=["Acquired"])

# Save datasets
train_df.to_csv("train.csv", index=False)
test_features1.to_csv("test1.csv", index=False)
test_features2.to_csv("test2.csv", index=False)
test_features3.to_csv("test3.csv", index=False)

# Save the actual answers (to be used for evaluation later)
test_answers = pd.concat([
    test_df1[["Company", "Acquired"]],
    test_df2[["Company", "Acquired"]],
    test_df3[["Company", "Acquired"]]
])
test_answers.to_csv("test_answers.csv", index=False)
