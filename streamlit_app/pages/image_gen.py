import streamlit as st
from st_clickable_images import clickable_images

import requests
import base64
from datetime import datetime

from google.cloud import storage
import google.auth
import google.auth.transport.requests


#st.set_page_config(layout="wide", page_title="Image search search")
#Change this path
root_dir = "/Users/shivajid/vertex_mlops_cicd/genAI/images/"
@st.cache_resource(show_spinner=False)
def _get_creds():
   creds, project = google.auth.default()
   creds.refresh(auth_req)
   auth_str = "Bearer " + creds.token
   return auth_str

samplecount = 8
if samplecount < 2: # minimum of 2 
    samplecount = 2
# creds.valid is False, and creds.token is None
# Need to refresh credentials to populate those

auth_req = google.auth.transport.requests.Request()
storage_client = storage.Client(project='demogct2022')
bucket_name= "vertex-gen-ai"

#print (creds.token)

def write_to_gcs(storage_client, bucket_name, blob_name, local_file_name):
    bucket= storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_file_name)






st.write("# Google Cloud Image Generation")
#col, buff2 = st.columns([2,4])
st.text_input("Seed",value=42,key="seed")
seed = st.session_state.seed
defstr = "A raccoon wearing formal clothes, wearing a top hat. Oil painting in the style of Vincent Van Gogh"
st.text_area("Enter your Prompt:", key="name", height=200, value=defstr)

content =  st.session_state.name
#print("Content", content)

if content is None or content == "":
    content = defstr
url = "https://us-central1-autopush-aiplatform.sandbox.googleapis.com/v1/projects/cloud-lvm-fishfooding/locations/us-central1/endpoints/6988653136307027968:predict"
json = '{"instances": [{"prompt":"' +  content+ '"}], "parameters": {"sampleImageSize": "256", "sampleCount": 8, "seed": "'+ seed+'" }}'

auth_str = _get_creds()
with st.spinner("Images are getting generated. Please wait....."):
 response = requests.post(url,json,
                         headers={'Authorization': auth_str, "Content-Type": "application/json" })


now = datetime.now()
now =  now.strftime("%m%d%Y%H%M%S")
#print(response.json())
#progress_text = "Image display is being loaded."
#my_bar = st.progress(0, text=progress_text)

i =0

img_grid =[]
img_ecoded=[]
if "predictions"  in response.json():
 for imgs  in response.json()["predictions"]:
    imgenc = imgs["bytesBase64Encoded"]
    img_ecoded.append(f"data:image/png;base64,{imgenc}" )
    img = base64.b64decode(imgenc)
    objname =  "GenImage_" + str(now)+ "__" +str(i)+".png"
    fname = root_dir + objname
    img_grid.append(fname)
    with open(fname, "wb") as f:
        f.write(img)
        write_to_gcs(storage_client, bucket_name, "generateimages/" + now +"/"+ objname,fname)
    #st.image(fname, caption=content, width=100,use_column_width="always")
    i=i+1
    gridcount = i-1 # as image index starts with 0
 st.write(i, " Images written to local file system and GCS Bucket: " + bucket_name)
 #st.write(" Following images generated in a 2 by 2 grid")
 
else:
   st.write("Error in response")
   st.write(response.json)
   

i=0
#gridcount
for rows in range(round(samplecount/2)):
  with st.container():
    col1, col2 = st.columns(2)
    with col1:
        #st.write('Col1 Caption for first chart')
        st.image(img_grid[gridcount])
        st.checkbox("Select Img" , key="my_var_" + str(i))
        gridcount = gridcount -1
        i=i+1
    with col2:
        #st.write('Col2 Caption for first chart')
        st.image(img_grid[gridcount])
        st.checkbox("Select Img", key="my_var_" + str(i))
        gridcount = gridcount -1
        i=i+1



    
