import json
import matplotlib.pyplot as plt

def loadTrainingHistory(*, filePath):
    f = open(filePath, 'r')
    trainingHistory = json.load(f)
    f.close()
    return trainingHistory

def plotLoss(history, savePath="./lossPlot.png"):
    epochs = [epoch["epoch"] for epoch in history]
    losses = [loss["loss"] for loss in history]

    plt.plot(epochs, losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.savefig(savePath)
    plt.close()

def plotAccuracy(history, savePath="./accuracyPlot.png"):
    epochs = [epoch["epoch"] for epoch in history]
    accuracies = [accuracy["accuracy"] for accuracy in history]

    plt.plot(epochs, accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Train accuracy (%)")
    plt.title("Training accuracy")
    plt.savefig(savePath)
    plt.close()

def generateTrainingPlots(*, filePath):
    history = loadTrainingHistory(filePath = filePath)
    plotLoss(history)
    plotAccuracy(history)