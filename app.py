import streamlit as st
from classifier import Classifier


@st.cache(allow_output_mutation=True)
def loadModel():
    return Classifier.getClassifier()

st.title("Twitter Sentiment Analysis")
st.markdown("#### A simple sentiment analyser of a sentence which analyse tone of the given sentence.")
classifierObj = loadModel()


tweet = st.text_input("Enter a sentence to analyze:", value="I love this project")
if len(tweet) > 0:
	output = classifierObj.predict(tweet)
	st.subheader("Prediction ✅")
	st.write(f"Label 🏷 : {output['label']}")
	st.write(f"Score 🤖 : {output['score']*100:0.2f}%")
	st.write(f"Elapsed Time ⌛: {output['elapsed_time']*100:0.2f} ms")
    
st.markdown("#### Made with :heart: by [Kaushik](https://github.com/ikaushikpal) and [Supriyo](https://www.github.com/supriyodam).")