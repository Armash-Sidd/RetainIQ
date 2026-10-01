import pandas as pd


def engineer_features(df):
    df = df.copy()

    df["IsZeroBalance"] = (
        df["Balance"] == 0
    ).astype(int)

    df["BalancePerProduct"] = (
        df["Balance"] / df["NumOfProducts"]
    )

    df["AgeSquared"] = df["Age"] ** 2

    df["ActiveProductInteraction"] = (
        df["IsActiveMember"] * df["NumOfProducts"]
    )

    df["AgeActivityInteraction"] = (
        df["Age"] * df["IsActiveMember"]
    )

    return df