import argparse

import logging
import requests
import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions
from  absl import  app
from absl import flags
import logging
import pandas as pd
import sys
from io import StringIO
from google.cloud import storage
import os

__INP_PATH = flags.DEFINE_string("input_path","gs://demogct/", "path for the file")
__OUT_PATH = flags.DEFINE_string("output_path","gs://demogct/","path for results")

class fileprocessor(beam.DoFn):

    def __init__(self):

        print ("In Init")

    def _get_creds(self):
        import google.auth
        import google.auth.transport.requests
        auth_req = google.auth.transport.requests.Request()
        creds, project = google.auth.default()
        creds.refresh(auth_req)
        auth_str = "Bearer " + creds.token
        return auth_str



    def get_request_url(slef):
            request_url = "https://us-vision.googleapis.com/v1/images:annotate?key=??"
            return request_url

    def get_req_json(self,image_path):
            value = '{"requests": [{"image": {"source": {"imageUri": "' + image_path + '"}},"features": [{"type": "IMAGE_EMBEDDING"}],"imageContext": {}}]}'
            print (value)
            return value

    def _get_response(self, image_path):
        import requests
        #print(auth_str)
        request_url = self.get_request_url()
        print(request_url)
        value = self.get_req_json(image_path)
        respObj = requests.post(request_url, value,
                                    headers={"Content-Type": "application/json"})


        return respObj

    def _get_embedding(self, respObj):
        emb = respObj.json()["responses"][0]["imageEmbeddingVector"]["imageEmbeddingVector"]
        return emb

    def _fetch_embeddings(self, bucket, image_url, img_name):
        print("In Fetch Embeddings")
        import os
        #auth_str = self._get_creds()
        respObj = self._get_response( image_url)
        print(respObj.status_code)
        if respObj.status_code == 200:
            with open(img_name, "w") as f:
                embedarr = self._get_embedding(respObj)
                print("Got Embeddings")

                f.write(str(embedarr))
                blob_name = "mjdata/embeddingv2/" + img_name
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(img_name)
            os.remove(img_name)


    def _fetch_images(self, bucket, image_url, image_name):
        import requests
        response = requests.get(image_url)
        img_name =  image_name
        if response.status_code == 200:
            with open(img_name, "wb") as f:
                f.write(response.content)
                blob_name = "mjdata/dfimages/" + img_name
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(img_name)
            os.remove(img_name)
        elif response.status_code == 429:
            self.sleep(10)
            self._fetch_images(self, bucket, image_url, image_name)
        else:
            logging.info("failed")

    def process(self, element, *args, **kwargs):
        #vals = element.split(",")
        #print(element)'
        import logging
        logging.info(element)
        imgurl = "dummy"
        from google.cloud import storage
        storage_client = storage.Client(project='demogct2022')
        bucket = storage_client.get_bucket("sd-img-data")
        import pandas as pd
        import sys
        from io import StringIO

        try:
         elemdata = StringIO(element)
         df = pd.read_csv(elemdata, header=None)
         rec= df.iloc[0]
         imgurl = rec[2]
         prompt = rec[1]
         image_name = rec[3]
         try:
          #self._fetch_images(bucket, imgurl, image_name)
          self._fetch_embeddings(bucket, imgurl, image_name)
          #print(f"Downloaded Embedding {imgurl}")
         except Exception as e:
             print("Failed Fetching FileException e", str(e))

        except Exception as e:
            print ("Error", str(e))

        #return imgurl



    def setup(self):
        print ("In setup")


def run(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        dest='input',
        default='gs://vertex-gen-ai/raw_data/Data_urls.csv',
        help='Input file to process.')
    parser.add_argument(
        '--output',
        dest='output',
        required=True,
        help='Output file to write results to.')
    logging.info("Executing run()")
    known_args, beam_args = parser.parse_known_args(argv)
    print(known_args, beam_args)
    pipeline_options=PipelineOptions(beam_args)
    # The pipeline will be run on exiting the with block.
    with beam.Pipeline(options=pipeline_options) as p:
        logging.info("Starting pipeline")
        lines = p | 'Read' >> ReadFromText("gs://vertex-gen-ai/raw_data/Data_urls.csv", skip_header_lines=True)
        step2 = lines | "Process Files" >> beam.ParDo(fileprocessor())
        #writefiles = step2 | 'Write' >> WriteToText("gs://vertex-gen-ai/mjdata/processed/")

    print ("Done!")

if __name__ == "__main__":
   run()
