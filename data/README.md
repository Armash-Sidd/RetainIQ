# Data Directory

This directory contains datasets used and produced by the RetainIQ customer churn prediction system.

## Directory Structure

```text
data/
├── raw/
│   └── bank_churn_raw.csv       # Primary raw customer records (immutable source)
├── processed/                   # Output folder for cleaned and train/test split data
└── README.md                    # Data dictionary and metadata documentation
```

## Dataset Description (`bank_churn_raw.csv`)

The raw dataset contains retail banking customer records with 14 attributes:

| Attribute | Data Type | Description |
| :--- | :--- | :--- |
| `id` | Integer | Internal record identifier |
| `CustomerId` | Integer | Unique identifier for bank client |
| `Surname` | String | Customer family name |
| `CreditScore` | Integer | Credit bureau score (300 to 850) |
| `Geography` | String | Country of residence (`France`, `Germany`, `Spain`) |
| `Gender` | String | Customer gender (`Male`, `Female`) |
| `Age` | Float / Int | Customer age in years |
| `Tenure` | Integer | Number of years as a customer (0 to 10) |
| `Balance` | Float | Current account balance |
| `NumOfProducts` | Integer | Number of bank products held (1 to 4) |
| `HasCrCard` | Binary (0/1) | Whether client holds a credit card |
| `IsActiveMember` | Binary (0/1) | Customer activity indicator |
| `EstimatedSalary` | Float | Estimated annual salary |
| **`Exited`** | **Binary (0/1)** | **Target variable**: `1` if closed account/churned, `0` if retained |

## Notes
- Files in `data/raw/` should be treated as read-only and immutable.
- Preprocessed splits and interim features will be stored in `data/processed/`.
