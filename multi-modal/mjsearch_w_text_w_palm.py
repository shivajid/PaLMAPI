import streamlit as st
import local_css 
import pandas as pd
import timeit
import requests
import re
import PIL


import vertexai
from vertexai.preview.language_models import TextGenerationModel,TextEmbeddingModel

import data
import torch
from models import imagebind_model
from models.imagebind_model import ModalityType

import pickle
import pinecone

from google.oauth2 import service_account
from google.cloud import aiplatform_v1beta1

icon = "rndcat.png"

st.set_page_config(layout="wide", page_title="multi modal search", page_icon=icon )

# Instantiate model

if "audiofname" not in st.session_state:
 st.session_state['audiofname'] = ""
if "inp_text" not in st.session_state:
 st.session_state["inp_text"] = ""


if "img" not in st.session_state:
 st.session_state["img"] = ""




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
 metadata_list_txt_uptd = ""
 with open('/home/admin_shivajid_altostrat_com/metadata_list_txt_uptd.jsonl', 'rb') as f:
  metadata_list_txt_uptd = pickle.load(f)
 return metadata_list_txt_uptd

def get_image_query_vector(device, model, image_list):
    sample_embedding = None
    sample_input = {
            ModalityType.VISION: data.load_and_transform_vision_data(image_list, device)
            }
    with torch.no_grad():
        sample_embedding = model(sample_input)
    test_vector = sample_embedding[ModalityType.VISION].cpu().detach().numpy()
    query_vector = test_vector[0].tolist()
    return query_vector

def get_audio_query_vector(device, model, audio_list):
    sample_embedding = None
    #st.write(audio_list)
    sample_input = {
        ModalityType.AUDIO: data.load_and_transform_audio_data(audio_list, device)
        }
    with torch.no_grad():
        sample_embedding =  model(sample_input)
    test_vector = sample_embedding[ModalityType.AUDIO].cpu().detach().numpy()
    query_vector = test_vector[0].tolist()
    return query_vector

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
def _get_text_resources():
 PROJECT_ID= "demogct2022"
 Region = "us-central1"
 vertexai.init(project=PROJECT_ID,location=Region)
 txtmodel = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
 return txtmodel

@st.cache_resource(show_spinner=False)
def get_resources():
 device, model = get_model()
 index = get_pnc_index()
 metadata = get_metadata()
 return  device, model, index, metadata

def _perform_search(inp, qtype="text"):
 device, model, index, metadata = get_resources()
 #print(f"In _perform search {qtype} {inp}")
 test_vector = []
 if qtype == "text":
  text_list = [inp]
  test_embedding = get_test_embedding(device, model,text_list)
  test_vector = _get_embeddings_array(test_embedding)
 elif qtype == "audio":
  audio_list = [inp]
  test_vector = get_audio_query_vector(device, model, audio_list)
 elif qtype == "image":
  image_list = [inp]
  test_vector = get_image_query_vector(device, model, image_list)
 match_ids = _get_match_ids(index, test_vector)
 results = []
 for idstr in match_ids:
    val = get_metadata_by_id(metadata, idstr)
    results.append(val)
 return test_vector, results

def _get_ann_config():
 sa_file_path = "demogct2022-ca8b4e443d11.json"
 scopes = ["https://www.googleapis.com/auth/cloud-platform"]
 sa_file_path = "/home/admin_shivajid_altostrat_com/ImageBind/demogct2022-ca8b4e443d11.json"
 credentials = service_account.Credentials.from_service_account_file(
      sa_file_path, scopes=scopes
  )
 client_options = {
      "api_endpoint": "547915122.us-central1-353713988661.vdb.vertexai.goog"
  }
 vertex_ai_client = aiplatform_v1beta1.MatchServiceClient(
      credentials=credentials,
      client_options=client_options,
  )
 request = aiplatform_v1beta1.FindNeighborsRequest(
      index_endpoint="projects/353713988661/locations/us-central1/indexEndpoints/4421646428682584064",
      deployed_index_id="mj_text_index_public_endpoint",
  )
 return request, vertex_ai_client

@st.cache_resource(show_spinner=False)
def get_cache_creds():
  # The AI Platform services require regional API endpoints.
  scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  
  # create a service account with `Vertex AI User` role granted in IAM page.
  # download the service account key https://developers.google.com/identity/protocols/oauth2/service-account#authorizingrequests
  sa_file_path = "/home/admin_shivajid_altostrat_com/ImageBind/demogct2022-ca8b4e443d11.json"
 
  credentials = service_account.Credentials.from_service_account_file(
      sa_file_path, scopes=scopes
  )
  client_options = {
      "api_endpoint": "1986786948.us-central1-353713988661.vdb.vertexai.goog"
       }
  return credentials, client_options






def findneighbor_sample(val):
  # The AI Platform services require regional API endpoints.
  #scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  
  # create a service account with `Vertex AI User` role granted in IAM page.
  # download the service account key https://developers.google.com/identity/protocols/oauth2/service-account#authorizingrequests
  #sa_file_path = "/home/admin_shivajid_altostrat_com/ImageBind/demogct2022-ca8b4e443d11.json"
 
  #credentials = service_account.Credentials.from_service_account_file(
  #    sa_file_path, scopes=scopes
  #)
  #client_options = {
  #    "api_endpoint": "1986786948.us-central1-353713988661.vdb.vertexai.goog"
  #}
  
  credentials, client_options = get_cache_creds()
  vertex_ai_client = aiplatform_v1beta1.MatchServiceClient(
      credentials=credentials,
      client_options=client_options,
  )
 
  request = aiplatform_v1beta1.FindNeighborsRequest(
      index_endpoint="projects/353713988661/locations/us-central1/indexEndpoints/7618076264208793600",
      deployed_index_id="mj_palm_text_index_public_endpoint_v2",
  )
  dp1 = aiplatform_v1beta1.IndexDatapoint(
      datapoint_id="0",
      feature_vector=val,
  )
  query = aiplatform_v1beta1.FindNeighborsRequest.Query(
      datapoint=dp1,
  )
  request.queries.append(query)
 
  response = vertex_ai_client.find_neighbors(request)
  return response


def imgbind_txt_findneighbor_sample( val):
  
 request, vertex_ai_client = _get_ann_config()
 dp1 = aiplatform_v1beta1.IndexDatapoint(
      datapoint_id="0",
      feature_vector=val,
 )
 query = aiplatform_v1beta1.FindNeighborsRequest.Query(
      datapoint=dp1,
 )
 request.queries.append(query)
 
 response = vertex_ai_client.find_neighbors(request)
 return response 

def get_match_id_embedding(matches):
 ids = []
 for neighbor in matches.nearest_neighbors:
  for neighbor in neighbor.neighbors:
    ids.append(neighbor.datapoint.datapoint_id)
 return ids

def get_matching_texts(ids):
 metadata = get_metadata()
 vals=[]
 for idstr in ids:
     val = get_metadata_by_id(metadata,idstr)
     vals.append(val)
 return vals

def _get_text_matches(val):
  resp = findneighbor_sample(val)
  ids = get_match_id_embedding(resp)
  #print(ids)
  vals = get_matching_texts(ids)
  return vals

def _get_imgbind_text_matches(val):
  resp = imgbind_txt_findneighbor_sample(val)
  ids = get_match_id_embedding(resp)
  #print(ids)
  vals = get_matching_texts(ids)
  return vals

def get_txt_embedding(val):
    model = _get_text_resources()
    query = ["machines of the future"]
    query = [val]
    embedarr = model.get_embeddings(query )
    return embedarr[0].values

def remove_urls(text):
  """Removes urls from text."""
  pattern = r'https?://\S+'
  text = re.sub(pattern, '', text)
  return text


@st.cache_resource(show_spinner=False)
def _get_audio_list():
    audiolist = ["",".assets/dog_audio.wav", ".assets/taunt.wav", ".assets/StarWars3.wav", ".assets/Chorus.wav", ".assets/Sports.wav", ".assets/Adver.wav",".assets/ImperialMarch60.wav",".assets/Pop1.wav",".assets/hindimusic.wav", ".assets/boneym.wav", ".assets/harley.wav",".assets/barack.wav" ]
    return audiolist

## Let's build the UI

colx, coly, colz = st.columns([0.1,0.5 , 0.2])
with colx:
    #st.write('')
    st.image('/home/admin_shivajid_altostrat_com/ImageBind/rndcat.png', width=180)
with coly:
    #st.warning('The images shown in this page are AI generated in Art form,collected from the internet. Art can be strange, calming, intriguing and disturbing.', icon="⚠️")
    
    #st.markdown("## Multi Modal Image Search  &nbsp;&nbsp;&nbsp;&nbsp;      [Embedding Map](https://atlas.nomic.ai/map/ae1ed3bf-abd3-4a42-9e90-7aacefbb94db/a14478fd-d527-41c4-9f7b-a0251e271d36)")
    st.markdown("## Multi Modal Search  &nbsp;&nbsp;&nbsp;&nbsp;")
  
    st.write("This is a multi modal search app based on Mid Journey's public dataset. The app is built on 114k of the 250K images. You can search for images and prompts using text description,audio clips and an image.") 
    #st.markdown("##### [Embedding map](https://atlas.nomic.ai/map/ae1ed3bf-abd3-4a42-9e90-7aacefbb94db/a14478fd-d527-41c4-9f7b-a0251e271d36)")
    #st.write(":exclamation: :red[There could be some disturbing Images as these are machine generated images by AI]")

#st.markdown("Enter a search string,describe the image you are looking for")
   
with coly:

    texttab, audiotab, imagetab = st.tabs(["Text Search", ":bold[ Audio Search]", "Image Search"])
    with texttab:
      input_txt = st.text_input(":silver[Search with Prompts]",  key = "search_txt", value="", help="Enter a text or idea. The text is used  search matching images on an image index. A second search for the matching prompts is done on prompt indexi.")
    with audiotab:
     audiolist = _get_audio_list()
     st.write("Select an audio clip and find images and text in the similar embedding space")
     fname = st.selectbox("Select and audio clip.", audiolist)
     if fname != "":
      st.audio(fname)
    with imagetab:
      uploadedfname = None
      uploaded_file = st.file_uploader("Choose a file", type=['png', 'jpg'])
      if uploaded_file is not None:
       uploadedfname = uploaded_file.name
       uploadedfname = f"tempdata/{uploadedfname}"
       bytes_data = uploaded_file.getvalue()
       with open(uploadedfname, "wb") as f:
          f.write(bytes_data)

with colz:
    placeholder = st.empty()

    #if uploadedfname != "" and uploadedfname != st.session_state["uploadedimage"]:
    #    placeholder.image(uploadedfname, width=200)

st.markdown("---")

cola, colb = st.columns([0.7,0.3])
#col1, col2, col3 = st.columns(3,gap="large")

if input_txt or fname != "" or uploadedfname:
 inpval = "" 
 modality = "text"
 #if input_txt:
 #    modality = "text"
 #    oldstate = 
 #    inpval = input_txt

 # elif (fname != "") and (oldfname != fname):
 #     modality = "audio"
 #     inpval = fname
 # else:
 #    modality = "text"
 #    inpval = input_txt

 oldfname = st.session_state['audiofname']
 oldinptext = st.session_state["inp_text"]
 olduploadedimage = st.session_state["img"]

 if (oldfname != fname):
     inpval = fname
     st.session_state["audiofname"] = fname
     modality = "audio"
     placeholder.empty()
 elif (oldinptext != input_txt):
     inpval = input_txt
     modality = "text"
     st.session_state["inp_text"] = input_txt
     placeholder.empty()
 elif uploadedfname and (olduploadedimage != uploadedfname):
     inpval = uploadedfname
     st.session_state["img"] = uploadedfname
     modality = "image"
     placeholder.image(uploadedfname, width=200)
 
 if inpval != "":
  test_vector, results = _perform_search(inpval, modality)

  with cola:
   col1, col2, col3 = st.columns(3)
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
 
  with colb:  
   st.write("&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; **Matching Prompts**")
   if modality == "text":
    text_emb = get_txt_embedding(input_txt)
    matching_texts = _get_text_matches(text_emb)
   else:
    #Call the IMG BIND Based Embedding Index in VME. As we are using for audio
    matching_texts = _get_imgbind_text_matches(test_vector)

   for val in matching_texts:
     img_url, prompt = get_images(val)
     #prompt = remove_urls(prompt)
     st.markdown(" ")
     st.text(prompt)


st.markdown("Developed by @Shivajid. Explore the Embedding [Map](https://atlas.nomic.ai/map/ae1ed3bf-abd3-4a42-9e90-7aacefbb94db/a14478fd-d527-41c4-9f7b-a0251e271d36) of the dataset on Nomic.ai.")
