import ml.models.base as ml

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import compute_class_weight

class RandomForest(ml.Base):
    def __init__(self, **kwargs):
        super().__init__()
        self.n_estimators = kwargs.get('n_estimators', 100)
        self.random_state = kwargs.get('random_state', 42)
        self.class_weight = kwargs.get('class_weight', 'balanced')
        self.model = None

    def fit(self, X, Y, **kwargs):
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            class_weight=self.class_weight
        )
        self.model.fit(X, Y)

    def predict(self, X, output_as_classifier=True, **kwargs):
        if self.model is None:
            raise UnboundLocalError("It is not possible to predict the data without training")
        
        predictions = self.model.predict(X)
        if output_as_classifier:
            predictions = predictions.astype(int)
        return predictions
