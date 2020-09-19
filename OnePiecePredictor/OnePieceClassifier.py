from OnePiecePredictor import *
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import sklearn
from sklearn.metrics import *
from category_encoders import TargetEncoder
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from .OnePiecePredictor import OnePiecePredictor
from sklearn.model_selection import train_test_split

class OnePieceClassifier(OnePiecePredictor):
    """
    This class can be used for hyper parameter tuning with cross validation and stratified splitting of data if required.

    X -> array-like(supported by Sklearn). If testTrainSplit is passed, this will be split into train and test
    Y -> array-like(supported by Sklearn). If testTrainSplit is passed, this will be split into train and test
    model -> string Currently supported models: LOGISTIC,RF,SVM,KNN,ADABOOST,XGBOOST,CATBOOST
    testX -> array-like(supported by Sklearn), test data. Ingnored if testTrainSplit is passed
    testY -> array-like(supported by Sklearn), test data. Ingnored if testTrainSplit is passed
    testTrainSplit -> float, ratio passed will be the amount of test data.
    stratify -> bool, used to perform stratified splitting. If passed data will be split based on Y.
    hyperParams -> dictionary, Hyper parameters specific to the model passed. If passed CV is performed.
    performCV -> bool, Used when hyperParams not passed to perform plain CV.
    folds -> int, No of folds to be used for CV.
    applySmote -> bool, To apply smote to oversample the data. Pass only one of applySmote or underSample
    underSample -> bool, To randomly undersample the majority data.
    sampling -> str, Values supported by SMOTE, RandomUnderSampler classes in imblearn library.
    scoring -> str, Evaluation metric. Currently supported values: accuracy,balanced_accuracy,f1,precision,recall,roc_auc. If not passed accuracy is used.
    targetEncodeCols -> List. List of columns to target encode.
    modelParams -> dictionary, Any model specific parameters can be passed as dictionary.
    multiClass -> Pass true in case of multiclass classification.
    """
    def __init__(self, X, Y, model ,testX = None, testY = None,testTrainSplit = None,
                stratify = None, hyperParams = None, performCV = None, folds = None,
                applySmote = False, underSample = False, sampling = None,
                scoring = None,  targetEncodeCols = None, multiClass = False, modelParams = {}):
        self.multiClass = multiClass
        self.applySmote = applySmote
        self.sampling = sampling
        self.stratify = stratify
        self.underSample = underSample
        super().__init__(
            X = X, Y = Y, testX=testX, testY=testY, testTrainSplit=testTrainSplit,
            model=model, folds=folds, hyperParams=hyperParams, scoring=scoring, performCV=performCV,
            targetEncodeCols = targetEncodeCols,
            modelParams = modelParams)
        if(not self.scoring):
            self.scoring = 'accuracy'
        self.scoreToFuncDict = self.__getScoreToFuncDict()

    def getTestTrainSlipt(self):
        ## If both testX and testTrainSplit are not passed throw exception.
        if ((self.testX is None) and (self.testTrainSplit is None)):
            raise Exception("Please pass testX or testTrainSplit")

        ## If targetEncodeCols is given first target encode them.
        if (self.targetEncodeCols):
            for col in self.targetEncodeCols:
                encoder = TargetEncoder()
                self.X[col] = encoder.fit_transform(self.X[col], self.Y)
                if(self.testX and self.testY):
                    self.testX[col] = encoder.fit_transform(self.testX[col], self.testY)

        if ((self.testX is not None) and (self.testTrainSplit is None)):
            return self.X, self.testX, self.Y, self.testY

        ## If stratify, smote and testTrainSplits are not passed, then just return.
        if (not self.stratify and not self.applySmote and not self.testTrainSplit):
            return self.X, self.testX, self.Y, self.testY

        # If startify flag is passed then stratify it using Y variable.
        startifyVar = self.Y if self.stratify else None
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, stratify=startifyVar,
                                                            test_size=self.testTrainSplit, random_state = 7)
        if (not self.applySmote and not self.underSample):
            return X_train, X_test, y_train, y_test
        else:
            if (self.applySmote):
                sm = SMOTE(sampling_strategy=self.sampling)
                X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
                return X_train_res, X_test, y_train_res, y_test
            if (self.underSample):
                underSampler = RandomUnderSampler(sampling_strategy=self.sampling)
                X_train_res, y_train_res = underSampler.fit_sample(X_train, y_train)
                return X_train_res, X_test, y_train_res, y_test

    def fit(self):
        super().fit()

    def test(self):
        super().test()

    def predict(self):
        if(self.testY is None):
            return 0, self.test()

        preds = self.bestEstimator.predict(self.testX)
        res = getattr(sklearn.metrics, self.scoreToFuncDict[self.scoring])(self.testY, preds)
        print(self.scoring, res)
        return res, preds

    def newDataPredict(self, testData):
        super().newDataPredict()

    def getEstimatorModel(self):

        specialMultiClassClassifiers = ["LOGISTIC"]
        modelToClassMapper = {
            "LOGISTIC" : LogisticRegression(),
            "RF" : RandomForestClassifier(),
            "SVM" : SVC(),
            "KNN" : KNeighborsClassifier(),
            "ADABOOST" : AdaBoostClassifier(),
            "XGBOOST" : XGBClassifier(),
            "CATBOOST" : CatBoostClassifier(),
        }

        if(self.model not in modelToClassMapper):
            raise Exception("Please pass a valid model from: LOGISTIC,RF,SVM,KNN,ADABOOST,XGBOOST,CATBOOST")
        else:
            cls = modelToClassMapper[self.model]
            if ((self.multiClass) and (self.model in specialMultiClassClassifiers) and (('multi_class' not in self.modelParams))):
                self.modelParams['multi_class'] = 'ovr'

            if ((self.model == "CATBOOST") and ('logging_level' not in self.modelParams)):
                self.modelParams['logging_level'] = 'Silent'

            return cls.set_params(**self.modelParams)


    def __getScoreToFuncDict(self):

        scoreToFuncDict = {
            "accuracy" : 'accuracy_score',
            "balanced_accuracy" : 'balanced_accuracy_score',
            "f1" : 'f1_score',
            "precision": 'precision_score',
            "recall" : 'recall_score',
            "roc_auc" : 'roc_auc_score'
        }

        if(self.scoring and self.scoring not in scoreToFuncDict):
            raise Exception("Please pass a valid scoring from: accuracy,balanced_accuracy,f1,precision,recall,roc_auc")

        return scoreToFuncDict

