from flask import Flask, render_template, request
from text_summary import summarizer
from means_cluster import summarizer1
from graphical import  summarizer2
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        rawtext = request.form['rawtext']
        summary, original_text, len_originaltext, len_summary, o, p = summarizer1(
            rawtext)
    return render_template('summary.html', summary=summary, original_text=original_text, len_original_text=len_originaltext, len_summary=len_summary, tsa=o, tss=p)

@app.route('/analyze1', methods=['GET', 'POST'])
def analyze1():
    if request.method == 'POST':
        rawtext = request.form['rawtext']
        summary, original_text, len_originaltext, len_summary, o, p = summarizer(
            rawtext)
    return render_template('summary.html', summary=summary, original_text=original_text, len_original_text=len_originaltext, len_summary=len_summary, tsa=o, tss=p)

@app.route('/analyze2', methods=['GET', 'POST'])
def analyze2():
    if request.method == 'POST':
        rawtext = request.form['rawtext']
        summary, original_text, len_originaltext, len_summary, o, p = summarizer(
            rawtext)
    return render_template('summary.html', summary=summary, original_text=original_text, len_original_text=len_originaltext, len_summary=len_summary, tsa=o, tss=p)

@app.route('/analyze3', methods=['GET', 'POST'])
def analyze3():
    if request.method == 'POST':
        rawtext = request.form['rawtext']
        summary, original_text, len_originaltext, len_summary, o, p = summarizer2(
            rawtext)
    return render_template('summary.html', summary=summary, original_text=original_text, len_original_text=len_originaltext, len_summary=len_summary, tsa=o, tss=p)


if __name__ == '__main__':
    app.run(debug=True)
