from flask import Flask, render_template, request
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Importing for logging purpose
import logging
from log_config import configure_logger

# Configure logger
configure_logger()

# Get logger
logger = logging.getLogger(__name__)

import os
from dotenv import load_dotenv

# Checking if the .env is loaded or not - Returns True
load_dotenv()

from huggingface_hub.hf_api import HfFolder

HfFolder.save_token('HUGGINGFACE_API_KEY')


Summarizationapp = Flask(__name__)
model_name = "sudhanshusinghaiml/google-flan-t5-base-fintuned"

@Summarizationapp.route('/health_check', methods = ['GET'])
def check():
    logger.info('This is inside health check function.')
    return "Yay! Flask App is running"

@Summarizationapp.route('/', methods = ['GET', 'POST'])
def index():
    if request.method == 'GET':
        logger.info('GET request method was triggered')
        return render_template('index.html', show_summary=False)

    elif request.method == 'POST':
        logger.info('POST request method was triggered')
        text = request.form['text_data']
        title = "Summary of the text"
        logger.info('Before calling function - summarize_the_text')
        summary = summarize_the_text(text)
        logger.info('After calling function - summarize_the_text')
        return render_template('summarize.html', summary=summary, text=text, show_summary=True, title=title)
    else:
        return "Http Method not recognized"


def summarize_the_text(article):

    logger.info('Inside the function - summarize_the_text')
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Tokenize and encode the article
    inputs = tokenizer(article, return_tensors='pt', max_length=1024, truncation=True)

    # Generate summary
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=4000, early_stopping=True)
    # summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    summary = " ".join([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])

    logger.info('Returning from - summarize_the_text')
    return summary

# 8. Run the API with uvicorn
#    Will run on http://127.0.0.1:5000
if __name__ == '__main__':
    Summarizationapp.run(host='0.0.0.0', debug=True)
    Summarizationapp.config['TEMPLATES_AUTO_RELOAD'] = True