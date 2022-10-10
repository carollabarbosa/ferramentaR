import pickle
import re
import string
from flask import Flask, jsonify, request
from flask import flash, redirect, render_template, request, session, abort
from flask_cors import CORS, cross_origin

punctuation_list = ["।", "”", "“", "’"]
for p in string.punctuation.lstrip():
    punctuation_list.append(p)

def prediction(txt):
    infile = open("D:/01 Research/Fake News/API/tfidf_char_pkl", 'rb')
    tfidf_char = pickle.load(infile)
    infile.close()
    x = tfidf_char.transform([txt])
    # print(x.shape)
    infile = open("D:/01 Research/Fake News/API/model", 'rb')
    clf = pickle.load(infile)
    infile.close()
    y_pred = clf.predict(x)
    print(y_pred)
    return y_pred[0]


def clean(doc):
    for p in punctuation_list:
        doc = doc.replace(p, "")
    doc = re.sub(r'[\u09E6-\u09EF]', "", doc, re.DEBUG)  # replace digits
    # doc = doc.replace("\n", "")

    return doc


# with open("input.txt", 'r') as infile:
#     doc = ""
#     for line in infile:
#         doc = doc + line.replace("\n", "")


# txt = clean(doc)
# prediction(txt)

app = Flask(__name__)
CORS(app)

# Build functionalities
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')
  
@app.route('/text')
def text():
   text= pd.read_csv('preprocessed05.csv')
   text_fake = 'text_falso'['label'].sum()
   text_true = 'text_verdadeiro'['label'].sum()
   resposta = {'text_verdadeiro': text_true}
   resposta = {'text_falso': text_fake}
   return jsonify (resposta)


@app.route("/", methods=['POST'])
def predict():

    doc = request.json['data']
    #doc = request.form["news"]
    print(doc)
    txt = clean(doc)

    infile = open("D:/01 Research/Fake News/API/tfidf_char_pkl", 'rb')
    tfidf_char = pickle.load(infile)
    infile.close()
    x = tfidf_char.transform([txt])
    # print(x.shape)
    infile = open("D:/01 Research/Fake News/API/model", 'rb')
    clf = pickle.load(infile)
    infile.close()
    y_pred = clf.predict(x)
    output = "Noticia Verdadeira"

    if y_pred == 0:
        output = "Noticia Falsa"
    print(y_pred)
    ret = '{"prediction":' + output + '}'

    #return render_template('form.html',value= output)
    return jsonify(result=output)


# running REST interface, port=5000 for direct test
if __name__ == "__main__":
    app.run(debug=False, host='127.0.0.1', port=4000)