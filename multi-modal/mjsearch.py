import streamlit as st
import pandas as pd
import timeit
import requests

import data
import torch
from models import imagebind_model
from models.imagebind_model import ModalityType

import pickle
import pinecone

# Instantiate model

@st.cache_resource(show_spinner=False)
def get_pnc_index():
 pinecone.init(api_key="b30849d2-2e1e-4d3a-a569-45841f556d40", environment="us-west4-gcp-free")
 index = pinecone.Index(pinecone.list_indexes()[0])
 return index

@st.cache_resource(show_spinner=False)
def get_model():
 device = "cuda:0" if torch.cuda.is_available() else "cpu"
 model = imagebind_model.imagebind_huge(pretrained=True)
 model.eval()
 model.to(device)
 return device, model

@st.cache_resource(show_spinner=False)
def get_metadata():
 metadata_list_txt_uptd = None
 with open('/home/admin_shivajid_altostrat_com/metadata_list_txt_uptd.jsonl', 'rb') as f:
  metadata_list_txt_uptd = pickle.load(f)
 return metadata_list_txt_uptd

def get_test_embedding(device, model, text_list):
 test_embeddings=None 

 test_inputs = {
    ModalityType.TEXT: data.load_and_transform_text(text_list, device)
   
  }

 with torch.no_grad():
    test_embeddings = model(test_inputs)
 return test_embeddings

def _get_embeddings_array(test_embedding):
 text_array = test_embedding["text"].cpu().detach().numpy()
 test_vector = text_array[0].tolist()
 return test_vector


def _get_match_ids(index, test_vector):
 query = index.query(
  vector=test_vector,
  top_k=12,
  include_values=False
 )
 match_ids = []
 for val in query["matches"]:
    match_ids.append(val["id"])
 return match_ids

def get_metadata_by_id(metadata_list_txt_uptd, idstr):
 val = metadata_list_txt_uptd[int(idstr)]
 return val

#def _fetch_image(img_url):


def get_images(val):
 img_url = val["url"]
 prompt = val["prompts"]
 return img_url, prompt

@st.cache_resource(show_spinner=False)
def get_resources():
 device, model = get_model()
 index = get_pnc_index()
 metadata = get_metadata()
 return device, model, index, metadata

def _perform_search(inp_text):
 device, model, index, metadata = get_resources()
 text_list = [inp_text]
 test_embedding = get_test_embedding(device, model,text_list)
 test_vector = _get_embeddings_array(test_embedding)
 match_ids = _get_match_ids(index, test_vector)
 results = []
 for idstr in match_ids:
    val = get_metadata_by_id(metadata, idstr)
    results.append(val)
 return results

## Let's build the UI
st.set_page_config(layout="wide")
colx, coly, colz = st.columns([0.1,0.7 , 0.2])
with colx:
    st.write('')
with coly:
    st.markdown("### Multi Modal Image Search        [Embedding Map](https://atlas.nomic.ai/map/ae1ed3bf-abd3-4a42-9e90-7aacefbb94db/a14478fd-d527-41c4-9f7b-a0251e271d36)")
    #st.markdown("##### [Embedding map](https://atlas.nomic.ai/map/ae1ed3bf-abd3-4a42-9e90-7aacefbb94db/a14478fd-d527-41c4-9f7b-a0251e271d36)")
    st.write(":exclamation: :red[There could be some disturbing Images as these are machine generated images by AI]")
#st.markdown("Enter a search string,describe the image you are looking for")
with colz:
    st.write("")
with coly:
    input_txt = st.text_input(":silver[Imagine....]", key = "search_txt", value="Machines of the future" )

st.markdown("---")
col1, col2, col3 = st.columns(3,gap="large")

if input_txt:
 results = _perform_search(input_txt)
 counter = 0
 for val  in results:
  img_url, prompt = get_images(val)
  #prompt= prompt.replace('[^a-zA-Z]', '')
  if counter in [0,3,6,9]:
   with col1:
      st.image(img_url )
  elif counter in [1,4,7,10]:
   with col2:
      st.image(img_url)
  elif counter in [2,5,8,11]:
   with col3:
      st.image(img_url)
  counter = counter +1


