import numpy as np
import pandas as pd


def EngineerFeatures(combined):
	combined["FundingPerEmployee"] = combined["FundingAmountUSD"] / (combined["EmployeeCount"] + 1)
	combined["AgeBucket"] = pd.cut(combined["Age"], bins=[0, 5, 10, 20, 100], labels=["0-5", "6-10", "11-20", "20+"])
	combined["FundingLevel"] = pd.qcut(combined["FundingAmountUSD"], q=4, labels=["Low", "Medium", "High", "Very High"])
	combined["LogFunding"] = np.log1p(combined["FundingAmountUSD"])
	combined["FundingEfficiency"] = combined["FundingAmountUSD"] / (combined["Age"] + 1)
	combined["IsBigCorp"] = ((combined["EmployeeCount"] > 1000) | (combined["FundingAmountUSD"] > 200_000_000)).astype(int)
	combined["IsStartup"] = (
    		(combined["Age"] <= 5) & 
    		(combined["EmployeeCount"] <= 50)
	).astype(int)

	#Analyzing Regional Risk to Expose Macroeconomic Features
	def region(loc):
    		if loc in ["US", "Canada"]:
        		return "NorthAmerica"
    		elif loc in ["India", "China", "Singapore"]:
        		return "Asia"
    		else:
        		return "Europe"

	combined["Region"] = combined["Location"].apply(region)
	combined = pd.get_dummies(combined, columns=["Region"], drop_first=True)
	combined["IsTech"] = combined["Industry"].isin(["AI", "FinTech", "Cybersecurity", "E-commerce", "Gaming"]).astype(int)
	combined["EmployeeBand"] = pd.cut(combined["EmployeeCount"], bins=[0, 50, 200, 1000, 10000],
                            labels=["Micro", "Small", "Mid", "Large"])
	combined = pd.get_dummies(combined, columns=["EmployeeBand"], drop_first=True)
	combined["YoungAndFunded"] = ((combined["Age"] <= 5) & (combined["FundingAmountUSD"] > 50_000_000)).astype(int)
	combined["EstimatedRevenue"] = combined["FundingAmountUSD"] * 0.2
	combined["EmployeesPerYear"] = combined["EmployeeCount"] / (combined["Age"] + 1)
	combined["IsGrowingFast"] = (combined["EmployeesPerYear"] > 50).astype(int)
	combined["IsHighRiskLocation"] = combined["Location"].isin(["China", "Russia"]).astype(int)
	combined["Overfunded"] = (combined["FundingAmountUSD"] > 300_000_000).astype(int)
	combined["InSweetSpot"] = (
    		(combined["Age"] >= 3) & (combined["Age"] <= 7) &
    		(combined["FundingAmountUSD"].between(10_000_000, 100_000_000)) &
    		(combined["EmployeeCount"].between(20, 200))
	).astype(int)
	combined["TalentDensity"] = combined["FundingAmountUSD"] / (combined["EmployeeCount"] + 1)
	combined["AcquisitionScore"] = (
    		0.4 * combined["FundingPerEmployee"].rank(pct=True) +
    		0.3 * combined["LogFunding"].rank(pct=True) +
    		0.3 * combined["IsStartup"]
	)
	hot_sectors = ["AI", "Cybersecurity", "HealthTech", "DevTools"]
	combined["InHotSector"] = combined["Industry"].isin(hot_sectors).astype(int)
	combined["CompetitorDensity"] = combined.groupby(["Industry", "Location"])["Company"].transform("count")
	combined["TooOldToBuy"] = (combined["Age"] > 15).astype(int)
	combined["CapitalEfficiency"] = combined["EstimatedRevenue"] / (combined["FundingAmountUSD"] + 1)
	combined["LastFundingYear"] = np.random.randint(2010, 2024, size=len(combined))
	combined["YearsSinceFunding"] = 2025 - combined["LastFundingYear"]
	combined["IPOReady"] = ((combined["Age"] > 5) & (combined["FundingAmountUSD"] > 50_000_000) & (combined["EmployeeCount"] > 200)).astype(int)
	combined["FundingPerYear"] = combined["FundingAmountUSD"] / (combined["Age"] + 1)
	combined["TeamLeverage"] = combined["FundingAmountUSD"] / (combined["EmployeeCount"] + 1)
	combined["LongLivedNoExit"] = ((combined["Age"] > 10) & (combined["Acquired"] == 0)).astype(int)
	combined["ExitPressure"] = ((combined["Age"] > 7) & (combined["FundingAmountUSD"] > 100_000_000)).astype(int)
	combined["Valuation"] = combined["FundingAmountUSD"] * np.random.uniform(1.5, 3.0, size=len(combined))
	combined["ValuationBand"] = pd.qcut(combined["Valuation"], 4, labels=["Low", "Medium", "High", "Very High"])

    #Label Encoding for Object values
    label_encoders = {}
    for col in df.select_dtypes(include='object'):
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col].astype(str))
        label_encoders[col] = le
    #Filling all NaN values
    combined = combined.fillna(combined.median(numeric_only=True))

	return combined