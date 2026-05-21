from tkinter import filedialog, Tk
import pandas as pd
import numpy as np


def loadFile(file_path):
    return pd.read_csv(file_path)

def dataToNumericalConverterForTargets(data):
    if data.isna().any().any():
        raise ValueError("Target column contains NaN values")

    data = data.to_numpy()
    target_class_size = len(set(data[:, 0].tolist()))
    unique_classes = sorted(set(data[:, 0].tolist()))
    class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}
    targetConfig = {"classToIndex": {cls: idx for idx, cls in enumerate(unique_classes)}}
    data = data.tolist()
    for row in range(len(data)):
        buffer = data[row][0]
        hot_index = class_to_index[buffer]
        data[row][0] = 0.0
        for i in range(target_class_size - 1):
            data[row].append(0.0)
        data[row][hot_index] = 1.0

    return np.array(data), target_class_size, targetConfig


def dataToNumericalConverter(data):
    if data.isna().any().any():
        raise ValueError("Feature columns contain NaN values")

    for col in data.select_dtypes(include=['object']).columns:
        asNum = pd.to_numeric(data[col], errors='coerce')
        if asNum.notna().all():
            data[col] = asNum

    numericCols = data.select_dtypes(include=['number', 'bool']).columns
    catCols = data.select_dtypes(include=['object', 'category']).columns

    featureConfig = {
        "numericCols": numericCols.tolist(),
        "catCols": catCols.tolist(),
        "minValues": {},
        "maxValues": {},
        "dummyColumns": []
    }

    parts = []
    if len(numericCols) > 0:
        dfNum = data[numericCols].astype(float)
        minVal = dfNum.min()
        maxVal = dfNum.max()
        featureConfig["minValues"] = minVal.to_dict()
        featureConfig["maxValues"] = maxVal.to_dict()
        dfNum = (dfNum - minVal) / (maxVal - minVal + 1e-8)
        parts.append(dfNum)
    if len(catCols) > 0:
        dfCat = pd.get_dummies(data[catCols], prefix=catCols).astype(float)
        featureConfig["dummyColumns"] = dfCat.columns.tolist()
        parts.append(dfCat)

    converted = pd.concat(parts, axis=1).to_numpy()
    return converted, converted.shape[1], featureConfig

def splitData(x, y, *, percent, seed=42):
    if len(x) != len(y):
        raise ValueError("x and y must have the same number of samples")
    if percent <= 0 or percent >= 100:
        raise ValueError("Test set percentage must be between 1-99")

    indices = np.arange(len(x))
    np.random.seed(seed)
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]
    xTest = x[0:int(len(x) * percent / 100)].copy()
    yTest = y[0:int(len(y) * percent / 100)].copy()
    xTrain = np.delete(x, slice(0, int(len(x) * percent / 100)), axis=0)
    yTrain = np.delete(y, slice(0, int(len(y) * percent / 100)), axis=0)

    return xTest, yTest, xTrain, yTrain

def loadAndPrepareData(*,filePath,targetColumn):
    df = loadFile(filePath)
    #dataClassColumn = df.pop(input(f"Enter the name of the column that represents the classes \n ({df.columns.to_list()}): "))
    dataClassColumn = df.pop(targetColumn)
    DataFrameClass = pd.DataFrame(dataClassColumn)
    x, data_size, featureConfig = dataToNumericalConverter(df)
    y, target_size, targetConfig = dataToNumericalConverterForTargets(DataFrameClass)

    return x, y, data_size, target_size, featureConfig, targetConfig


