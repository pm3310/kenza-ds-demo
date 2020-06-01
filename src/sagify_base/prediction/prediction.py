import os


_MODEL_PATH = os.path.join('/opt/ml/', 'model')  # Path where all your model(s) live in


class ModelService(object):
    model = None

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model is None:
            # TODO Load a specific model
            # TODO Examples:
            # TODO 1. keras.models.load_model(os.path.join(_MODEL_PATH, '<model_file>'))
            # TODO 2. joblib.load('<model_file>')
            from sklearn.externals import joblib
            cls.model = joblib.load(os.path.join(_MODEL_PATH, 'model.pkl'))
        return cls.model

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them."""
        clf = cls.get_model()
        return clf.predict(input)


def predict(json_input):
    """
    Prediction given the request input
    :param json_input: [dict], request input
    :return: [dict], prediction
    """

    model_input = json_input['features']
    prediction = ModelService.predict(model_input)

    return {
        "prediction": prediction.item()
    }
