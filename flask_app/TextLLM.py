'''
This is a sample app to invoke the TextLLM Pages
Please install all the needed libraries 
@Author: Shivaji Dutta
@Date: 05/16/2023
'''

import pandas as pd
import timeit

import vertexai
from vertexai.preview.language_models import TextGenerationModel

#Set Project Variables
PROJECT_ID= ""
model_name = "text-bison@001"
location = "us-central1"

vertexai.init(project=PROJECT_ID, location=location)
model = TextGenerationModel.from_pretrained(model_name)



def predict_large_language_model_sample(
    temperature: float,
    max_decode_steps: int,
    top_p: float,
    top_k: int,
    content: str,
    tuned_model_name: str = "",
    ) :
    """Predict using a Large Language Model."""
    response = model.predict(
        content,
        temperature=temperature,
        max_output_tokens=max_decode_steps,
        top_k=top_k,
        top_p=top_p,)
    return response





