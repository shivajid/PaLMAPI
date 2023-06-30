######################################
# This module queries the matching engine
# returns the matching indexes
# Appends them to a query
# logs the results to a BigTable
# Author: Shivaji Dutta
# Date: 05/19/2023
#####################################

import streamlit as st
#from google.cloud.aiplatform.private_preview.language_models import  TextGenerationResponse, TextEmbeddingModel
import vertexai
from vertexai.preview.language_models import TextGenerationModel
from vertexai.preview.language_models import TextEmbeddingModel
from streamlit_chat import message
import pandas
from google.cloud import bigtable
from google.cloud import happybase
import timeit
from google.oauth2 import service_account
from google.cloud import aiplatform_v1beta1

from datetime import datetime

icon = "Wikipedia-logo-v2@2x.png"
st.set_page_config(page_title="Wikipedia search", page_icon=icon )
col1, col2, col3 = st.columns(3)
with col2:
 st.image("https://ichef.bbci.co.uk/news/976/cpsprodpb/12DE1/production/_123218277_gettyimages-1368388509.jpg", width=250)
#st.write("xxxThe Corpus is on 2022 Winter Olympics Wikipedia Scraped Articles. The Application is powered by PaLM TextGeneration, Embedding, Matching Enging and BigTable")
#st.write("The response is slow, please have patience as generating text takes time...")
st.write("Ask a question on 2022 Winter Olympics")


@st.cache_resource(show_spinner=False)
def get_project_details():
 PROJECT_ID = "demogct2022" #@param
 INSTANCE_ID = "bus-instance" #@param
 TABLE_ID = "masterdata" #@param
 COLUMN_FAMILY_NAME = "cf1" #@param
 return PROJECT_ID, INSTANCE_ID, TABLE_ID, COLUMN_FAMILY_NAME

@st.cache_resource(show_spinner=False)
def get_BigTable_table():
 PROJECT_ID, INSTANCE_ID, TABLE_ID, COLUMN_FAMILY_NAME = get_project_details()
 client = bigtable.Client(project=PROJECT_ID, admin=True)
 instance = client.instance(INSTANCE_ID)
 connection = happybase.Connection(instance=instance)
 table = connection.table(TABLE_ID)
 column_family_name = "cf1"
 column_name = "{fam}:content".format(fam=column_family_name)
 return table, column_name

#Bigtable prompt logger
@st.cache_resource(show_spinner=False)
def get_logging_table():
 plog_INSTANCE_ID = "inferresults"
 plog_TABLE_ID = "promptlogger"
 plog_COLUMN_FAMILY_NAME = "cf1" 
 #appname = "WikipediaAskv2.py"
 PROJECT_ID = "demogct2022" 
 #Connect to Prompt Logger Table
 pclient = bigtable.Client(project=PROJECT_ID, admin=True)
 pinstance = pclient.instance(plog_INSTANCE_ID)
 pconnection = happybase.Connection(instance=pinstance)
 ptable = pconnection.table(plog_TABLE_ID)
 pcolumn_family_name = plog_COLUMN_FAMILY_NAME
 pcolumn_name = "{fam}:prompt".format(fam=pcolumn_family_name)
 rcolumn_name = "{fam}:resp".format(fam=pcolumn_family_name)
 return ptable, pcolumn_name, rcolumn_name

@st.cache_resource(show_spinner=False)
def get_text_embedding_model():
 model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
 return model


def _findneighbor_sample( val,
                         api_endpoint="563741590.us-central1-353713988661.vdb.vertexai.goog",
                         index_endpoint="projects/353713988661/locations/us-central1/indexEndpoints/2315298809212567552",
                         deployed_index = "wikipedia_olympics_2022_QA"
                         ):
  # The AI Platform services require regional API endpoints.
  scopes = ["https://www.googleapis.com/auth/cloud-platform"]

  # create a service account with `Vertex AI User` role granted in IAM page.
  # download the service account key https://developers.google.com/identity/protocols/oauth2/service-account#authorizingrequests
  sa_file_path = "/home/admin_shivajid_altostrat_com/pages/demogct2022-ca8b4e443d11.json"

  credentials = service_account.Credentials.from_service_account_file(
      sa_file_path, scopes=scopes
  )
  client_options = {
      "api_endpoint": "563741590.us-central1-353713988661.vdb.vertexai.goog"
  }

  vertex_ai_client = aiplatform_v1beta1.MatchServiceClient(
      credentials=credentials,
      client_options=client_options,
  )

  request = aiplatform_v1beta1.FindNeighborsRequest(
      index_endpoint="projects/353713988661/locations/us-central1/indexEndpoints/2315298809212567552",
      deployed_index_id="wikipedia_ann_public_endpoint",
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

import json


def query_embedding(
    query: str
):
 model = get_text_embedding_model()
 return model.get_embeddings([query])


def fetch_vertex_matches(emb):

   matches = _findneighbor_sample (emb)
   return matches

def strings_ranked_by_relatedness(query):
  pass

def get_row_key(index, max_size):
   rkey = "wikipedia_string" + "#" + str(index).zfill(max_size)
   return rkey

#Prompt Logger Utilities
def get_date_time():
   return  datetime.today().strftime('%Y-%m-%d#%H:%M:%S')

def get_prow_key():
   #rkey = "wikipedia_string" + "#" + str(index).zfill(max_size)
   dt = get_date_time()
   appname = "WikipediaAskv2.py"
   key_prefix = f"prompt_logger#{appname}"
   rkey = key_prefix + "#" + dt
   return rkey

def putTable(table, pcolumn_name, prompt, rcolumn_name, resp ):
  row_key = get_prow_key()
  table.put(row_key, {pcolumn_name.encode("utf-8"): prompt.encode("utf-8"), rcolumn_name.encode("utf-8"): resp.encode("utf-8")})
  print ("Data Entered")
  return row_key





# Invoke the LLM model

@st.cache_resource(show_spinner=False)
def get_textgen_model(PROJECT_ID, location, model_name):
  vertexai.init(project=PROJECT_ID, location=location)
  model = TextGenerationModel.from_pretrained(model_name)
  return model

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
    model = get_textgen_model(project_id, location, model_name)
    if tuned_model_name:
      model = model.get_tuned_model(tuned_model_name)
    response = model.predict(
        content,
        temperature=temperature,
        max_output_tokens=max_decode_steps,
        top_k=top_k,
        top_p=top_p,)
    #print(f"Response from Model: {response.text}")
    return response




max_size = 4
def lookup_text_by_id(id):
 table, column_name =get_BigTable_table()
 rkey = get_row_key(id, max_size)
 b_rkey=rkey.encode("utf-8")
 row = table.row(b_rkey)
 return row[b'cf1:content'].decode("utf-8")



def get_match_id_embedding(val):
 ids = []
 matches = _findneighbor_sample(val[0].values)
 for neighbor in matches.nearest_neighbors:
  for neighbor in neighbor.neighbors:
    #print(neighbor.distance)
    #print(neighbor.datapoint.datapoint_id)
    ids.append(neighbor.datapoint.datapoint_id)
 return ids


def get_prompt_text_from_matches(match_ids) :
  prompt_text = ""
  for match_id in match_ids:
    prompt_text = prompt_text + " " + lookup_text_by_id(match_id)
  return prompt_text

def ask( Question,tok):
  #start_time = timeit.default_timer()

  val = query_embedding(Question)
  #elapsed = timeit.default_timer() - start_time
  #print ("Elapsed Time query_embedding:", elapsed)

  #start_time = timeit.default_timer()
  match_ids = get_match_id_embedding(val)
  #elapsed = timeit.default_timer() - start_time
  #print ("Elapsed Time match_ids:", elapsed)

  #start_time = timeit.default_timer()
  prompt_text = get_prompt_text_from_matches(match_ids[:5])
  #elapsed = timeit.default_timer() - start_time
  #print ("Elapsed Time prompt_text:", elapsed)

  prompt = "Based on the text below, " + Question+ "\n context:" + prompt_text
  predictions_text =predict_large_language_model_sample(
      "demogct2022", "text-bison@latest", 0.2, tok, 0.8, 40,
          prompt,
       "us-central1")
  
  elapsed = timeit.default_timer() - start_time
  #print ("Elapsed Time prediction_text:", elapsed)
  return predictions_text

#ask("When city wass Irina Avvakumova born?")
#Question = "List the winners of the 2022 Winter Olympics: "
Question = ""
st.text_area("Enter your Question:", key="name", height=200, value=Question)
with st.sidebar:
    st.slider("Max Tokens", min_value=128, max_value=1024,key="tok", step=16, value=512)

content =  st.session_state.name
tok = st.session_state.tok

if content != "":
    #content = Question
    with st.spinner("PROCESSING. Please wait....."):
     start_time = timeit.default_timer()
     response = ask(content, tok)
     elapsed2 = timeit.default_timer() - start_time
     st.write("Elapsed Time:", elapsed2)
     st.write(response)
     if response is not None or response.text != "": 
      ptable, pcolumn_name, rcolumn_name = get_logging_table()
      putTable(ptable, pcolumn_name, content, rcolumn_name, response.text)



