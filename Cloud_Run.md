# This file is for hosting on the cloud

## Create repo in Artifact Repository
```
REPOSITORY = "Palm-repo"
gcloud artifacts repositories create $REPOSITORY --repository-format=docker --location=us-central1 --description="PALM Streamllit App"
```
Get details of your repo
```
gcloud artifacts repositories describe palm-streamlit
```

## Authorize 

```
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
```

## Build the Dockerfile

Here we build a docker image with name "palm-stremlit", as it is going to be the main app.
```
docker build --tag palm-streamlit .
```

Check the image name
```
docker images
```


## Retag to your image
```
PROJECT_ID = <your project id>
docker tag palm-streamlit us-central1-docker.pkg.dev/$PROJECT_ID/palm-streamlit/palm-streamlit:latest
```

## Push the docker image to the Artifact Registry

```
 docker push us-central1-docker.pkg.dev/$PROJECT_ID/palm-streamlit/palm-streamlit
 ```
 
 ## Deploy your Docker image to Cloud Run
 
 Note the dockerfile has a port of 8503. You can run it on any port. Change the port number
 
 List the images to check it is there
 ```
 gcloud artifacts docker images list us-central1-docker.pkg.dev/demogct2022/palm-streamlit/palm-streamlit 
 ```
 **Deploy**
 
 Set Iam role
 ```
 gcloud beta run services add-iam-policy-binding --region=us-central1 --member=allUsers --role=roles/run.invoker palm-demo
 ```
 
Change the port number to port you would like to use
 ```
 gcloud run deploy palm-demo --image us-central1-docker.pkg.dev/$PROJECT/palm-streamlit/palm-streamlit --region=us-central1 --port=8503  --allow-unauthenticated
 ```
 
 Check the console for any Errors
 
 **Success**
 On Success you should see the following output
 ```
 Service [palm-demo] revision [palm-demo-00005-zus] has been deployed and is serving 100 percent of traffic.
Service URL: https://palm-demo-s5ag53ldnq-uc.a.run.app
```

 
 
 
 

