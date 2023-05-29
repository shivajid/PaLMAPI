This folder contains two Files
 -  TextLLM_GDev_PaLM.py
 -  Full_Enterprise_Search.py


## TextLLM_GDev_PaLM
This is a simple streamlit example to work with the PaLM API from developer.google.com

Please get an api key from developer.google.com and update the file.
```
palm.configure(api_key='')
``` 

This should give you and interactive API to work with the tool

## Full_Enterprise_Search.py

This is a simple web app to wprk with Enterprise Search

Please get the follwing details from your ES Index in GCP and update the file.

```#Set these variable from your search engine
es_project_id = ""
es_location = ""                    
search_engine_id = ""
serving_config_id = "" 
```
You can get the above values from the API Integrations Page
