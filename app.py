from llm.llm import LLM
from flask import Flask, render_template, request


app = Flask(__name__)
llm_instance = LLM()

@app.route('/', methods=['GET', 'POST'])
def index():
    response = None

    if request.method == 'POST':
        txt_input = request.form['txt_input']
        response = llm_instance.generate(txt_input)

    return render_template('index.html', response=response)

if __name__ == '__main__':
    app.run(debug=True)
