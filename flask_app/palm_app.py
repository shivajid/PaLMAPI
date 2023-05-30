from flask import render_template
from flask import request
from flask import render_template
from markupsafe import escape
from flask import Flask
from . import TextLLM as tp

#Enter your project id
PROJECT_ID=""
temp=0.9
max_tok=256
top_p = 0.2
top_k=0.5



app = Flask(__name__, template_folder='templates')



@app.route("/")
def defualt():
    return render_template("landing.html")

@app.route('/palm', methods=['GET'])
def get_page():
    return render_template("landing.html")

def _get_response(content):
 resp = tp.predict_large_language_model_sample(PROJECT_ID, "text-bison@001", temp, max_tok, top_p, top_k,  content, "us-central1")
 return resp


@app.route('/process', methods=['GET', 'POST'])
def process():
    print("In process")
    prompt ="Who is the president of USA?"
    prompt = request.args.get("prompt")
    print ("--:"+prompt + ":--")
    if prompt != "":
     ans = _get_response(prompt)
    else:
     ans = "No Prompt Entered. Please try again!!"
    print(ans)
    return f"<a>{ans}</a>"
