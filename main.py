import convertersModule as cM
from network import Network
import plotsModule as pM
import argparse

def training_network(*,networkParametersCLI,filePath):
    x, y, data_size, target_size, featureConfig, targetConfig = cM.loadAndPrepareData(filePath=filePath,targetColumn=networkParametersCLI["targetColumn"])
    xTest, yTest, xTrain, yTrain= cM.splitData(x, y, percent=networkParametersCLI["testSplitPercent"])
    model = Network(dataSize=data_size, targetSize=target_size,NetworkParametersCLI = networkParametersCLI)
    print(model)
    model.trainNetwork(epochs = networkParametersCLI["epochs"], x=xTrain, y=yTrain, lr=networkParametersCLI["learningRate"], stopCondition = networkParametersCLI["targetTrainAccuracyPercent"])
    model.evaluateNetwork(x=xTest, y=yTest)
    model.saveNetworkParameters(saveMethod="npz")
    model.saveNetworkTrainingHistory()
    pM.generateTrainingPlots(filePath = "./NetworkTrainingHistory.json")
    print(featureConfig)

def read_data_from_file(*,filePath):
    x, y, data_size, target_size, featureConfig, targetConfig = cM.loadAndPrepareData(filePath=filePath,targetColumn=networkParametersCLI["targetColumn"])
    model = Network(dataSize=data_size, targetSize=target_size, WeightsConfFilePath="./NetworkWeights.npz", BiasesConfFilePath="./NetworkBiases.npz", ParametersConfFilePath="./NetworkParameters.json")
    print(model)
    model.evaluateNetwork(x=x, y=y)

def define_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True,metavar="{train,eval}")

    parser_train = subparsers.add_parser("train", help="Train the network")

    parser_train.add_argument("-ln","--layers_num",required=True, type=int, help="The number of layers in the network")
    parser_train.add_argument("-af","--activation_function",required=True, type=str, choices=["relu", "softmax", "tanh", "sigmoid"],action="extend",help="Activation function for each layer in the network (tanh,sigmoid,relu,softmax)", nargs="+")
    parser_train.add_argument("-nn","--neurons_number", type=int, help="Number of neurons for each layer in the network",action="extend", nargs="+",required=True)
    parser_train.add_argument("-lf","--loss_function", type=str,choices=["mse", "crossentropy"], help="Loss function in the network",required=True)
    parser_train.add_argument("-lr","--learning_rate",type=float,required=True,help="Learning rate for the network")
    parser_train.add_argument("-e","--epochs",type=int,required=True,help="Number of epochs for the network")
    parser_train.add_argument("-ttap","--target_train_accuracy_percent",type=float,required=True,help="Stop training when the model reaches this accuracy on the training set. Must be between 0 and 100. For example, 60 means training stops after reaching at least 60 percent training accuracy")
    parser_train.add_argument("-tsp","--test_split_percent",type=float,required=True,help="Percentage of the dataset used for testing. For example, 30 means 30 percent of the data will be used for testing, and the remaining 70 percent will be used for training")
    parser_train.add_argument("-fp","--file_path",help="Path to the dataset file",required=True,type=str)
    parser_train.add_argument("-tc", "--target_column",help="Target column name from dataset",required=True,type=str)

    parser_eval = subparsers.add_parser("eval", help="Evaluate the network")

    parser_eval.add_argument("-fp","--file_path",help="Path to the dataset file",required=True,type=str)
    parser_eval.add_argument("-tc", "--target_column", help="Target column name from dataset", required=True, type=str)

    return parser

if __name__ == "__main__":
    parser = define_parser()
    args = parser.parse_args()
    if args.mode == "train":
        if len(args.activation_function) != args.layers_num:
            parser.error("The number of activation functions must match the number of layers")
        if len(args.neurons_number) != args.layers_num:
            parser.error("Number of neuron values does not match number of layers in the network")

        networkParametersCLI = {
            "layersNum": args.layers_num,
            "layers": {f"layer_{index}": (neuronsNum, activation_function) for index, (neuronsNum, activation_function)
                       in enumerate(zip(args.neurons_number, args.activation_function), start=1)},
            "lossFunction": args.loss_function,
            "learningRate": args.learning_rate,
            "epochs": args.epochs,
            "targetTrainAccuracyPercent": args.target_train_accuracy_percent,
            "testSplitPercent": args.test_split_percent,
            "targetColumn": args.target_column,
        }

        training_network(networkParametersCLI = networkParametersCLI,filePath=args.file_path)
    elif args.mode == "eval":
        read_data_from_file(filePath=args.file_path)
