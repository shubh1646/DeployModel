from flask import Flask, render_template, redirect, request, jsonify
from sklearn.externals import joblib 
app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/")
def hello():
   return "hello world"


@app.route("/a",methods=['POST'])
def marks():
    if request.method == 'POST':
        data = request.json
        print(data["time"])
        hours  = float(data["time"])
       # hours = float(request.form["hours"])
        marks = model.predict([[hours]])[0][0]
        fmarks  = str(round(marks,1))
        output = {'marks': fmarks}
        
    return output
    #return render_template("index.html",your_marks = fmarks)

if __name__ == '__main__':
    app.run(debug= True)
 