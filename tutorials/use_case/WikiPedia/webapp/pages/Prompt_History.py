
from datetime import datetime
from google.cloud import bigtable
from google.cloud import happybase
import pandas as pd
import streamlit as st



#Bigtable
@st.cache_resource(show_spinner=False)
def _get_Big_Table():
  PROJECT_ID="demogct2022"#@param
  plog_INSTANCE_ID = "inferresults" #@param
  plog_TABLE_ID = "promptlogger" #@param
  plog_COLUMN_FAMILY_NAME = "cf1" #@param
  appname = "TextLLM"#@param

  client = bigtable.Client(project=PROJECT_ID, admin=True)
  instance = client.instance(plog_INSTANCE_ID)
  connection = happybase.Connection(instance=instance)
  table = connection.table(plog_TABLE_ID)
  column_family_name = plog_COLUMN_FAMILY_NAME
  pcolumn_name = "{fam}:prompt".format(fam=column_family_name)
  rcolumn_name = "{fam}:resp".format(fam=column_family_name)
  return table,pcolumn_name, rcolumn_name

def get_date_time():
   return  datetime.today().strftime('%Y-%m-%d#%H:%M:%S')

def get_row_key():
   #rkey = "wikipedia_string" + "#" + str(index).zfill(max_size)
   dt = get_date_time()
   key_prefix = f"prompt_logger#{appname}"
   rkey = key_prefix + "#" + dt
   return rkey

def putTable(table, pcolumn_name, prompt, rcolumn_name, resp ):
  row_key = get_row_key()

  table.put(row_key, {pcolumn_name.encode("utf-8"): prompt.encode("utf-8"), rcolumn_name.encode("utf-8"): resp.encode("utf-8")})
  print ("Data Entered")
  return row_key

def _print_rows():
 patt = "prompt_logger#WikipediaAskv2".encode("utf-8")
 table, pcolumn_name, rcolumn_name = _get_Big_Table()
 row = table.scan(row_prefix=patt)

 df = pd.DataFrame()
 parray = []
 resparray = []

 for key, data in row:
  text = data[b'cf1:prompt'].decode("utf-8")
  resp = data[b'cf1:resp'].decode("utf-8")
  #print(text)
  #print(resp)
  #print("*" * 80)
  parray.append(text)
  resparray.append(resp)

 df["prompt"] = parray
 df["response"] = resparray
 Data_reverse_row_1 = df.iloc[::-1]
 st.table(Data_reverse_row_1)

_print_rows()
