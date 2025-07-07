import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def EngineerFeatures(df, is_train=True):
    df = df.copy()

    df["FundingPerEmployee"] = df["FundingAmountUSD"] / (df["EmployeeCount"] + 1)
    df["AgeBucket"] = pd.cut(df["Age"], bins=[0, 5, 10, 20, 100], labels=["0-5", "6-10", "11-20", "20+"])
    df["FundingLevel"] = pd.qcut(df["FundingAmountUSD"], q=4, labels=[0, 1, 2, 3])
    df["LogFunding"] = np.log1p(df["FundingAmountUSD"])
    df["FundingEfficiency"] = df["FundingAmountUSD"] / (df["Age"] + 1)

    df["IsBigCorp"] = ((df["EmployeeCount"] > 1000) | (df["FundingAmountUSD"] > 200_000_000)).astype(int)
    df["IsStartup"] = ((df["Age"] <= 5) & (df["EmployeeCount"] <= 50)).astype(int)

    def region(loc):
        if loc in ["US", "Canada"]:
            return "NorthAmerica"
        elif loc in ["India", "China", "Singapore"]:
            return "Asia"
        else:
            return "Europe"
    
    df["Region"] = df["Location"].apply(region)
    df = pd.get_dummies(df, columns=["Region"], drop_first=True)

    df["IsTech"] = df["Industry"].isin(["AI", "FinTech", "Cybersecurity", "E-commerce", "Gaming"]).astype(int)
    df["EmployeeBand"] = pd.cut(df["EmployeeCount"], bins=[0, 50, 200, 1000, 10000], labels=["Micro", "Small", "Mid", "Large"])
    df = pd.get_dummies(df, columns=["EmployeeBand"], drop_first=True)

    df["YoungAndFunded"] = ((df["Age"] <= 5) & (df["FundingAmountUSD"] > 50_000_000)).astype(int)
    df["EstimatedRevenue"] = df["FundingAmountUSD"] * 0.2
    df["EmployeesPerYear"] = df["EmployeeCount"] / (df["Age"] + 1)
    df["IsGrowingFast"] = (df["EmployeesPerYear"] > 50).astype(int)
    df["IsHighRiskLocation"] = df["Location"].isin(["China", "Russia"]).astype(int)
    df["Overfunded"] = (df["FundingAmountUSD"] > 300_000_000).astype(int)

    df["InSweetSpot"] = (
        (df["Age"] >= 3) & (df["Age"] <= 7) &
        (df["FundingAmountUSD"].between(10_000_000, 100_000_000)) &
        (df["EmployeeCount"].between(20, 200))
    ).astype(int)

    df["TalentDensity"] = df["FundingAmountUSD"] / (df["EmployeeCount"] + 1)
    df["AcquisitionScore"] = (
        0.4 * df["FundingPerEmployee"].rank(pct=True) +
        0.3 * df["LogFunding"].rank(pct=True) +
        0.3 * df["IsStartup"]
    )

    hot_sectors = ["AI", "Cybersecurity", "HealthTech", "DevTools"]
    df["InHotSector"] = df["Industry"].isin(hot_sectors).astype(int)
    df["TooOldToBuy"] = (df["Age"] > 15).astype(int)
    df["CapitalEfficiency"] = df["EstimatedRevenue"] / (df["FundingAmountUSD"] + 1)

    df["LastFundingYear"] = np.random.randint(2010, 2024, size=len(df))
    df["YearsSinceFunding"] = 2025 - df["LastFundingYear"]

    df["IPOReady"] = ((df["Age"] > 5) & (df["FundingAmountUSD"] > 50_000_000) & (df["EmployeeCount"] > 200)).astype(int)
    df["FundingPerYear"] = df["FundingAmountUSD"] / (df["Age"] + 1)
    df["TeamLeverage"] = df["FundingAmountUSD"] / (df["EmployeeCount"] + 1)

    if is_train and "Acquired" in df.columns:
        df["LongLivedNoExit"] = ((df["Age"] > 10) & (df["Acquired"] == 0)).astype(int)
    else:
        df["LongLivedNoExit"] = 0

    df["ExitPressure"] = ((df["Age"] > 7) & (df["FundingAmountUSD"] > 100_000_000)).astype(int)
    df["Valuation"] = df["FundingAmountUSD"] * np.random.uniform(1.5, 3.0, size=len(df))
    df["ValuationBand"] = pd.qcut(df["Valuation"], 4, labels=["Low", "Medium", "High", "Very High"])

    label_encoders = {}
    for col in df.select_dtypes(include='object'):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    for col in ["AgeBucket", "ValuationBand"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    df = df.drop(columns=["Company"], errors='ignore')
    df = df.fillna(df.median(numeric_only=True))
    df = df.astype("float32")

    return df
