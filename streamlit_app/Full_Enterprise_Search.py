import streamlit as st
import re
import json


es_project_id = "ucs-fishfood-6"
es_location = "global"                    
search_engine_id = "alphapet_1684629621521"
serving_config_id = "default_config"          
search_query = "What is Google's revenue in 2022?"


from google.cloud import discoveryengine_v1beta as genappbuilder

search_engine_id = "alphapet_1684629621521"
serving_config_id ="default_search_widget_config"


def search_sample(
    project_id: str,
    location: str,
    search_engine_id: str,
    serving_config_id: str,
    search_query: str,
) -> None:
    # Create a client
    client = genappbuilder.SearchServiceClient()

    # The full resource name of the search engine serving config
    # e.g. projects/{project_id}/locations/{location}
    serving_config = client.serving_config_path(
        project=project_id,
        location=location,
        data_store=search_engine_id,
        serving_config=serving_config_id,
    )

    request = genappbuilder.SearchRequest(
        serving_config=serving_config,
        query=search_query,
    )
    results = []
    response_pager = client.search(request)
    response = genappbuilder.SearchResponse(
        results=response_pager.results,
        facets=response_pager.facets,
        guided_search_result=response_pager.guided_search_result,
        total_size=response_pager.total_size,
        attribution_token=response_pager.attribution_token,
        next_page_token=response_pager.next_page_token,
        corrected_query=response_pager.corrected_query,
        summary=response_pager.summary,
    )

    response_json = genappbuilder.SearchResponse.to_json(
        response, including_default_value_fields=False, indent=2
    )
    for result in response.results:
        #print(result)
        results.append(result)
    return results, response, response_json

def build_response(response_json):
   val= json.loads(response_json)
   st.write("Search Results: \n")
   st.write("Total Pages :",val["totalSize"])
   st.markdown("---")
   for data in val["results"][:5]:
    link = data["document"]["derivedStructData"]["link"]
    
    try:
      snip = data["document"]["derivedStructData"]["snippets"]
      #print ("Snippets ",data["document"]["derivedStructData"]["snippets"])
      for snippet in snip:
       st.markdown(snippet["snippet"])
       st.write("Page: ",snippet["pageNumber"], " of ", link)
    except:
     pass
    st.markdown("---")
    st.write("\n")

st.header("Google Earnings Question & Answer:")
defaultstr = "Welcome to Google Cloud LLM"
st.text_area("Enter your Prompt:", key="name", height=200, value=search_query)

search_query =  st.session_state.name
content = search_query
print("Content", content)
if content is None or content == "":
    content = search_query


with st.spinner("Please wait.. Enterprise Search at works!!"):
  results, response,  response_json = search_sample(es_project_id,es_location, search_engine_id, serving_config_id,search_query)
  #st.write("Response from Enterprise Search")
  st.markdown("---")
  st.write("Summary:")
  st.write("\n")
  st.markdown(response.summary.summary_text)
  st.markdown("---")
  st.write("\n\n")
  build_response(response_json)
