import joblib
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time


physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except Exception as e:
    print(e)


class Classifier:
    SENTIMENT_THRESHOLDS = (0.4, 0.7)
    instance = None

    def __init__(self, model_path='model/early_weights.hdf5',
                 word_tokenizer_path='model/word_tokenizer.h5'):
        self.model = load_model(model_path)
        self.word_tokenizer = joblib.load(word_tokenizer_path)

    @classmethod
    def getClassifier(cls):
    	if cls.instance is None:
    		newInstance = cls()
    		cls.instance = newInstance
    	return cls.instance
    	 
    def decode_sentiment(self, score, include_neutral=True):
        if include_neutral:        
            label = 'Neutral 😐'
            if score <= Classifier.SENTIMENT_THRESHOLDS[0]:
                label = 'Negative 😠'
            elif score >= Classifier.SENTIMENT_THRESHOLDS[1]:
                label = 'Positive 😄'

            return label
        else:
            return 'Negative 😠' if score < 0.5 else 'Positive 😄'

    def predict(self, text, include_neutral=True):
        start_at = time.time()
        # Tokenize text
        x_test = pad_sequences(self.word_tokenizer.texts_to_sequences([text]), maxlen=30)
        # Predict
        score = self.model.predict([x_test])[0]
        # Decode sentiment
        label = self.decode_sentiment(score, include_neutral=include_neutral)

        return {"label": label,
                "score": float(score),
                "elapsed_time": time.time()-start_at
                }  
