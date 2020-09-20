import abc
from onepiecepredictor.MultiModelsPredictor import *
from onepiecepredictor.OnePieceClassifier import *

class MultiModelsClassifier(MultiModelsPredictor):
    """
        For Comparing multiple classification models performance with cross validation and stratified splitting of data if required.

        X -> array-like(supported by Sklearn). If testTrainSplit is passed, this will be split into train and test
        Y -> array-like(supported by Sklearn). If testTrainSplit is passed, this will be split into train and test
        testX -> array-like(supported by Sklearn), test data. Ignored if testTrainSplit is passed
        testY -> array-like(supported by Sklearn), test data. Ignored if testTrainSplit is passed
        testTrainSplit -> float, ratio passed will be the amount of test data.
        stratify -> bool, used to perform stratified splitting. If passed data will be split based on Y.
        performCV -> bool, Used when hyperParams not passed to perform plain CV.
        folds -> int, No of folds to be used for CV.
        applySmote -> bool, To apply smote to oversample the data. Pass only one of applySmote or underSample
        underSample -> bool, To randomly undersample the majority data.
        sampling -> str, Values supported by SMOTE, RandomUnderSampler classes in imblearn library.
        scoring -> str, Evaluation metric. Currently supported values: accuracy,balanced_accuracy,f1,precision,recall,roc_auc. If not passed accuracy is used.
        targetEncodeCols -> List. List of columns to target encode.
        multiClass -> Pass true in case of multiclass classification.
    """

    def __init__(self, X, Y, testX = None, testY = None,testTrainSplit = None,
                 folds = 5, scoring = None, performCV = None, targetEncodeCols = None,
                 applySmote=False, underSample=False, sampling=None, stratify=None, multiClass = False
                 ):
        self.multiClass = multiClass
        self.applySmote = applySmote
        self.sampling = sampling
        self.stratify = stratify
        self.underSample = underSample
        super().__init__(X=X, Y=Y, testX=testX, testY=testY, testTrainSplit=testTrainSplit,
                         folds=folds, scoring=scoring, performCV=performCV, targetEncodeCols=targetEncodeCols
                         )

    def predict(self):
        """
        Returns dictionary with keys as Models and Values as metric scores.
        """
        dummyRef = OnePieceClassifier(X = self.X,  Y = self.Y, model = "LOGISTIC", modelParams = {},testTrainSplit = self.testTrainSplit,
                                    testX = self.testX, testY = self.testY,folds = self.folds, scoring = self.scoring, performCV = self.performCV,
                                    targetEncodeCols = self.targetEncodeCols, applySmote = self.applySmote, underSample = self.underSample,
                                    sampling = self.sampling, stratify = self.stratify, multiClass = self.multiClass)

        tempX = dummyRef.trainX
        tempY = dummyRef.trainY
        tempTestX = dummyRef.testX
        tempTestY = dummyRef.testY

        classifiers = ["LOGISTIC","RF","SVM","KNN","ADABOOST","XGBOOST","CATBOOST"]
        resultsDict = {}
        for classifier in classifiers:
            op = OnePieceClassifier(X = tempX,  Y = tempY, model = classifier, modelParams = {}, testTrainSplit = None,
                                    testX = tempTestX, testY = tempTestY, folds = self.folds, scoring = self.scoring, performCV = self.performCV)

            op.fit()
            score, preds = op.predict()
            resultsDict[classifier] = score

            del op

        print(resultsDict)



