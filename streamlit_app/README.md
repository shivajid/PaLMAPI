# Vertex PaLM API
This repo has examples of interacting with the PaLM API. 
I use the python streamlit library to interact with the API.

## Install

Install the code dependencies

```
pip3 install -r requirements.txt
```

## Authenticate

```
gcloud auth application-default login
```
## Update PROJECT_ID

Edit the TaskLLM.py file to update the PROJECT_ID
```
PROJECT_ID= "demogct2022"
```


## Run the App


```
streamlit run TextLLM.py
```

You should see something like this:

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://10.0.0.72:8501
```
Navigate to the URL to test the app!!

Have Fun!!

## Troubleshooting

You may see an error with protobuf. If you try and uninstall and reinstall streamlit, the issue goes away. There is a clash in the protobuf library that is shipped with google cloud and streamlit.


