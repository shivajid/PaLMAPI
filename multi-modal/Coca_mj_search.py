import typing
import streamlit as st
import pandas as pd
import timeit
import requests
import re
import PIL
import base64
from google.cloud import storage
from google.cloud import aiplatform
from google.protobuf import struct_pb2
from streamlit_card import card

import vertexai
from vertexai.preview.language_models import TextGenerationModel,TextEmbeddingModel

import pickle
from google.oauth2 import service_account
from google.cloud import aiplatform_v1beta1

icon = "rndjoker.png"

st.set_page_config(layout="wide", page_title="multi modal search", page_icon=icon )

if "prompt_text" not in st.session_state:
 st.session_state['prompt_text'] = ""
if "inp_text" not in st.session_state:
 st.session_state["inp_text"] = ""

if "img" not in st.session_state:
 st.session_state["img"] = ""

@st.cache_resource(show_spinner=False)
def loadllist():
  namelist = []
  with open("allnamelist.pkl", "rb") as f:
    namelist = pickle.load(f)
  return namelist
 
@st.cache_resource(show_spinner=False)
def load_url_mapping_list():
  orig_df = pd.read_csv("gs://vertex-gen-ai/raw_data/Data_urls.csv")
  return orig_df
 

@st.cache_resource(show_spinner=False)
def _get_text_resources():
 PROJECT_ID= "demogct2022"
 Region = "us-central1"
 vertexai.init(project=PROJECT_ID,location=Region)
 txtmodel = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
 return txtmodel

@st.cache_resource(show_spinner=False)
def get_Embedding_resources():
  PROJECT_ID = 'demogct2022' 
  location = "us-central1"
  api_regional_endpoint = "us-central1-aiplatform.googleapis.com"
  client_options = {"api_endpoint": api_regional_endpoint}
  client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
  location = location
  project = PROJECT_ID
  return client, project, location


def get_embedding( text : str = None, image_bytes : bytes = None):
       # Inspired from https://stackoverflow.com/questions/34269772/type-hints-in-namedtuple.
     class EmbeddingResponse(typing.NamedTuple):
      text_embedding: typing.Sequence[float]
      image_embedding: typing.Sequence[float]
     client, project, location = get_Embedding_resources()
     if not text and not image_bytes:
       raise ValueError('At least one of text or image_bytes must be specified.')

     instance = struct_pb2.Struct()
     if text:
       instance.fields['text'].string_value = text

     if image_bytes:
       encoded_content = base64.b64encode(image_bytes).decode("utf-8")
       image_struct = instance.fields['image'].struct_value
       image_struct.fields['bytesBase64Encoded'].string_value = encoded_content

     instances = [instance]
     endpoint = (f"projects/{project}/locations/{location}"
       "/publishers/google/models/multimodalembedding@001")
     response = client.predict(endpoint=endpoint, instances=instances)

     text_embedding = None
     if text:
       text_emb_value = response.predictions[0]['textEmbedding']
       text_embedding = [v for v in text_emb_value]

     image_embedding = None
     if image_bytes:
       image_emb_value = response.predictions[0]['imageEmbedding']
       image_embedding = [v for v in image_emb_value]

     return EmbeddingResponse(
       text_embedding=text_embedding,
       image_embedding=image_embedding)

# Extract image embedding
def getImageEmbeddingFromImageContent(content):
  response = get_embedding(text=None, image_bytes=content)
  return response.image_embedding

def getImageEmbeddingFromFile(filePath):
  with open(filePath, "rb") as f:
    return getImageEmbeddingFromImageContent(f.read())

# Extract text embedding
def getTextEmbedding(text):
  response = get_embedding(text=text, image_bytes=None)
  return response.text_embedding


def match_img_url(imgname):
 orig_df=load_url_mapping_list()
 val = orig_df[orig_df["name"] == imgname]
 img_url = val.iloc[0]["urls"]
 promptval = val.iloc[0]["prompts"]
 return promptval, img_url

def parse_name(imgname):
 imgname = imgname.split("/")[-1]
 return imgname



def findneighbor_coca_sample(val):
  # The AI Platform services require regional API endpoints.
  scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  
  # create a service account with `Vertex AI User` role granted in IAM page.
  # download the service account key https://developers.google.com/identity/protocols/oauth2/service-account#authorizingrequests
  sa_file_path = "/home/admin_shivajid_altostrat_com/ImageBind/demogct2022-ca8b4e443d11.json"
 
  credentials = service_account.Credentials.from_service_account_file(
      sa_file_path, scopes=scopes
  )
  client_options = {
      "api_endpoint": "2068783116.us-central1-353713988661.vdb.vertexai.goog"
  }
 
  vertex_ai_client = aiplatform_v1beta1.MatchServiceClient(
      credentials=credentials,
      client_options=client_options,
  )
 
  request = aiplatform_v1beta1.FindNeighborsRequest(
      index_endpoint="projects/353713988661/locations/us-central1/indexEndpoints/3589043447572463616",
      deployed_index_id="mj_coca_img_index_public_endpoint_v2",
  )
  dp1 = aiplatform_v1beta1.IndexDatapoint(
      datapoint_id="0",
      feature_vector=val,
  )
  query = aiplatform_v1beta1.FindNeighborsRequest.Query(
      datapoint=dp1,
      neighbor_count=15
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

def get_matching_img_urls(matchids):
 match_img_urls = []
 prompt_match = []
 namelist = loadllist()
 for id in matchids:
  fname = namelist[int(id)]
  cleanedfname = parse_name(fname)
  prompts, img_url = match_img_url(cleanedfname)
  match_img_urls.append(img_url)
  prompt_match.append(prompts)
 return prompt_match,match_img_urls


def match_by_text(usertext):
 emb = getTextEmbedding(usertext)
 respObj = findneighbor_coca_sample(emb)
 matchids = get_match_id_embedding(respObj)
 prompt_match, img_urls = get_matching_img_urls(matchids)
 return prompt_match, img_urls

def match_by_img(imgfname):
 emb = getImageEmbeddingFromFile(imgfname)
 respObj = findneighbor_coca_sample(emb)
 matchids = get_match_id_embedding(respObj)
 prompt_match,img_urls = get_matching_img_urls(matchids)
 return prompt_match, img_urls

def _perform_search(inpval, modality):
  img_urls = []
  prompt_match =[]
  if modality == "text":
    prompt_match, img_urls = match_by_text(inpval)
  else:
    prompt_match, img_urls = match_by_img(inpval)
  return prompt_match, img_urls

#Palm Text Embedding API

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
      neighbor_count=15
  )
  request.queries.append(query)
 
  response = vertex_ai_client.find_neighbors(request)
  return response

@st.cache_resource(show_spinner=False)
def get_metadata():
 metadata_list_txt_uptd = ""
 with open('/home/admin_shivajid_altostrat_com/metadata_list_txt_uptd.jsonl', 'rb') as f:
  metadata_list_txt_uptd = pickle.load(f)
 return metadata_list_txt_uptd

def get_metadata_by_id(metadata_list_txt_uptd, idstr):
 val = metadata_list_txt_uptd[int(idstr)]
 return val

def get_matching_texts(ids):
 metadata = get_metadata()
 vals=[]
 for idstr in ids:
     val = get_metadata_by_id(metadata,idstr)
     vals.append(val)
 return vals

def get_txt_embedding(val):
    model = _get_text_resources()
    query = ["machines of the future"]
    query = [val]
    embedarr = model.get_embeddings(query )
    return embedarr[0].values

def _get_text_matches(val):
  resp = findneighbor_sample(val)
  ids = get_match_id_embedding(resp)
  #print(ids)
  vals = get_matching_texts(ids)
  return vals

def get_images(val):
 img_url = val["url"]
 prompt = val["prompts"]
 return img_url, prompt


# Let's Build the UI
colx, coly, colz = st.columns([0.1,0.5 , 0.2])

with colx:
    st.image('/home/admin_shivajid_altostrat_com/ImageBind/rndjoker.png', width=180)
with coly:
    st.markdown("## Generative AI Image Search  &nbsp;&nbsp;&nbsp;&nbsp;")
  
   
with coly:

    texttab,  imagetab, promptTab = st.tabs(["Text Search",  "Image Search", "Prompt Search"])
    with texttab:
      input_txt = st.text_input(":silver[Search with Prompts]",  key = "search_txt", value="", help="Enter a text or idea. The text is used  search matching images on an image index. A second search for the matching prompts is done on prompt indexi.")
    with imagetab:
      uploadedfname = None
      uploaded_file = st.file_uploader("Choose a file", type=['png', 'jpg'])
      if uploaded_file is not None:
       uploadedfname = uploaded_file.name
       uploadedfname = f"tempdata/{uploadedfname}"
       bytes_data = uploaded_file.getvalue()
       with open(uploadedfname, "wb") as f:
          f.write(bytes_data)
    with promptTab:
      prompt_txt = st.text_input(":silver[Search for Prompts]",  key = "prompt_txt", value="", help="Search for prompts to help build better prompts")

with colz:
    placeholder = st.empty()

st.markdown("---")
#Image column and text column
cola, colb = st.columns([0.7,0.3])
if input_txt or prompt_txt or  uploadedfname:
 inpval = "" 
 modality = "text"
 oldinptext = st.session_state["inp_text"]
 olduploadedimage = st.session_state["img"]
 oldprompttext = st.session_state["prompt_text"]
 if (oldinptext != input_txt):
     inpval = input_txt
     modality = "text"
     st.session_state["inp_text"] = input_txt
     placeholder.empty()
 elif uploadedfname and (olduploadedimage != uploadedfname):
     inpval = uploadedfname
     st.session_state["img"] = uploadedfname
     modality = "image"
     placeholder.image(uploadedfname, width=200)
 elif (oldprompttext != prompt_txt):
    modality = "prompt"

 if inpval != "" and modality != "prompt":
  col1, col2, col3 = st.columns(3)
  prompt_match, results = _perform_search(inpval, modality)
  with cola:
   counter = 0
   for img_url  in results:
     #print(img_url)
     if counter in [0,3,6,9]:
      with col1:
        st.image(img_url , width=400)
        #hasclicked =card(
        #    title="",
        #    text="",
        #    image=img_url,
        #     url=""
        #    )
        st.markdown(prompt_match[counter])
     elif counter in [1,4,7,10]:
      with col2:
        st.image(img_url, width=400)
        st.markdown(prompt_match[counter])
     elif counter in [2,5,8,11]:
      with col3:
        st.image(img_url, width=400)
        st.markdown(prompt_match[counter])
     counter = counter +1
 elif modality == "prompt":
     st.write("&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; **Matching Prompts**")
    # print("Embedding Generation - text ",prompt_txt)
     col1, col2, col3 = st.columns(3)
     text_emb = get_txt_embedding(prompt_txt)
     matching_texts = _get_text_matches(text_emb)
     counterx = 0
     for val in matching_texts:
      img_url, prompt = get_images(val)
      #prompt = remove_urls(prompt)
    
      #for img_url  in results:
        #print(img_url)
      if counterx in [0,3,6,9]:
          with col1:
           st.markdown(f"<div><span>{prompt}</span></div>", unsafe_allow_html=True)
           st.image(img_url , width=400)
      elif counterx in [1,4,7,10]:
          with col2:
           st.markdown(f"<div><span>{prompt}</span></div>", unsafe_allow_html=True)
           st.image(img_url, width=400)
           
      elif counterx in [2,5,8,11]:
          with col3:
            st.markdown(f"<div><span>{prompt}</span></div>", unsafe_allow_html=True)
            st.image(img_url, width=400)
      counterx = counterx +1
      #st.markdown(" ")
      #st.image(img_url)
      
      #st.markdown("---")

st.markdown("Developed by @Shivajid. Explore the Embedding [Map](https://atlas.nomic.ai/map/ae1ed3bf-abd3-4a42-9e90-7aacefbb94db/a14478fd-d527-41c4-9f7b-a0251e271d36) of the dataset on Nomic.ai.")




  
