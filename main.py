import convertersModule as cM
from network import Network
import plotsModule as pM

def training_network():
    x, y, data_size, target_size, featureConfig, targetConfig = cM.loadAndPrepareData()
    xTest, yTest, xTrain, yTrain= cM.splitData(x, y, percent=50)
    model = Network(layersNum=3, dataSize=data_size, targetSize=target_size)
    print(model)
    model.trainNetwork(epochs = 5000, x=xTrain, y=yTrain, lr=0.1, stopCondition = 60)
    model.evaluateNetwork(x=xTest, y=yTest)
    model.saveNetworkParameters(saveMethod="npz")
    model.saveNetworkTrainingHistory()
    pM.generateTrainingPlots(filePath = "./NetworkTrainingHistory.json")
    print(featureConfig)
    print(targetConfig)

def read_data_from_file():
    x, y, data_size, target_size, featureConfig, targetConfig = cM.loadAndPrepareData()
    xTest, yTest, xTrain, yTrain = cM.splitData(x, y, percent=30)
    model = Network(dataSize=data_size, targetSize=target_size, WeightsConfFilePath="./NetworkWeights.npz", BiasesConfFilePath="./NetworkBiases.npz", ParametersConfFilePath="./NetworkParameters.json")
    print(model)
    model.evaluateNetwork(x=xTest, y=yTest)


choice = input("Do you want to train the network? (Y/n): ").strip().lower()
if choice in ("y", ""):
    training_network()
else:
    read_data_from_file()
