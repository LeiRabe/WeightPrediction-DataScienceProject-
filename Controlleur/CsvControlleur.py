import pandas as pd


class CsvControlleur:
    def __init__(self):
        self.csvPath = "../predictions/datasets_26073_33239_weight-height.csv"

    def readCsv(self):
        # Load data
        dataset = pd.read_csv(self.csvPath)

        # Convert data
        # inches to cm
        height = dataset["Height"].tolist()
        height_cm = []
        for h in height:
            h *= 2.54
            height_cm.append(h)
        dataset["Height"] = height_cm
        # lbs to kg
        weight = dataset["Weight"].tolist()
        weight_kg = []
        for w in weight:
            w *= 0.453592
            weight_kg.append(w)
        dataset["Weight"] = weight_kg

        # Convert Gender to number
        dataset['Gender'].replace('Female', 0, inplace=True)
        dataset['Gender'].replace('Male', 1, inplace=True)

        return dataset