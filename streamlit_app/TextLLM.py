'''
This is a sample app to invoke the TextLLM Pages
Please install all the needed libraries 
'''

import streamlit as st
import pandas as pd
import timeit

import vertexai
from vertexai.preview.language_models import TextGenerationModel

#Set Project Variables
PROJECT_ID= "demogct2022"
model_name = "text-bison@001"
location = "us-central1"




def predict_large_language_model_sample(
    project_id: str,
    model_name: str,
    temperature: float,
    max_decode_steps: int,
    top_p: float,
    top_k: int,
    content: str,
    location: str = "us-central1",
    tuned_model_name: str = "",
    ) :
    """Predict using a Large Language Model."""
    vertexai.init(project=project_id, location=location)
    model = TextGenerationModel.from_pretrained(model_name)
    if tuned_model_name:
      model = model.get_tuned_model(tuned_model_name)
    response = model.predict(
        content,
        temperature=temperature,
        max_output_tokens=max_decode_steps,
        top_k=top_k,
        top_p=top_p,)
    return response


#Build the UI Skeleton
defaultstr = "How to build safe portfolio on Etsy"
st.text_area("Enter your Prompt:", key="name", height=200, value=defaultstr)

with st.sidebar:
  st.slider("Temperature",min_value=0.0,max_value=1.0,step=0.01, key="temp", value = 0.2)
  st.slider("TopK",min_value=1,max_value=40,step=1, key="topk", value=25)
  st.slider("TopP", min_value=0.0, max_value=1.0, key="topp", value=0.95, step=0.01)
  st.slider("Max Tokens", min_value=128, max_value=1024,key="tok", step=16, value=1024)


content =  st.session_state.name
top_p = st.session_state.topp
top_k = st.session_state.topk
max_tok = st.session_state.tok
temp = st.session_state.temp


if content is None or content == "":
 content = defaultstr

with st.spinner("Please wait.. AI at works!!"):
   start_time = timeit.default_timer()
   resp = predict_large_language_model_sample("demogct2022", "text-bison@001", temp, max_tok, top_p, top_k,  content, "us-central1")
   elapsed = timeit.default_timer() - start_time

st.write("Elapsed time:", round(elapsed,2), " seconds")
predictions = resp.text

#for prediction in predictions:
#    print(" prediction:", prediction["content"])
#    st.write(prediction["content"])
st.write(predictions)

