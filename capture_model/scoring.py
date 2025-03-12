import sys
sys.path.append('/home/ec2-user/environment/capture_model_old-master')


import os
import logging
from typing import List
from capture_model.constants import *
from capture_model.modelling import FasttextClassifierModel, PreProcessor
from capture_model import post_processing as postproc

capture_models_dir = os.path.dirname(__file__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLiogger(__name__)

package_data_dir = os.path.join(capture_models_dir, 'package_data')
line_classifier = FasttextClassifierModel.from_json(os.path.join(package_data_dir, 'model_fasttext.json'))
preprocessor = PreProcessor.from_json(os.path.join(package_data_dir, 'preproc.json'))

def classify_lines(lines: List[str]):
    """Classifies multiple lines using only FastText."""
    classified_lines = []
    
    for line in lines:
        try:
            classified_line = classify_line(line)
            classified_lines.append(classified_line)
            logger.debug(str(classified_line))
        except Exception as e:
            logger.exception(f"Exception while classifying line '{line}': \n {e}")

    return classified_lines

def classify_line(line: str):
    """Classifies a single line using only FastText."""
    prediction_item = predict_line(line)

    line_cat = prediction_item['category_id']
    line_score = prediction_item['category_score']
    line_text = prediction_item['line']

    line_class = 'TEXT'
    line_value = ''
    line_text_value = ''

    prices = postproc.find_prices(line_text)
    price_count = len(prices)
    cvr = postproc.find_cvr(line_text)
    cvr_count = len(cvr)
    date = postproc.parse_date(line_text)

    if price_count == 1:
        line_class = 'PRICE'
        line_value = prices[0].strip()
    elif price_count > 1:
        line_class = 'MULTI_PRICE'
        line_value = prices

    if line_cat == TOTAL_CATEGORY and price_count == 1 and line_score > TOTAL_CUTOFF:
        line_class = 'TOTAL'
    elif line_cat == DISCOUNT_CATEGORY and price_count >= 1 and line_score > DISCOUNT_CUTOFF:
        line_class = 'DISCOUNT'
    elif line_cat == MOMS_CATEGORY and price_count == 1 and line_score > MOMS_CUTOFF:
        line_class = 'MOMS'
    elif line_cat == CVR_CATEGORY and cvr_count >= 1 and line_score > CVR_CUTOFF:
        line_class = 'CVR'
    elif line_cat == CHANGE_CATEGORY and price_count == 1 and line_score > CHANGE_CUTOFF:
        line_class = 'CHANGE'
    elif line_cat == MEMBERSHIP_CATEGORY:
        line_class = 'MEMBERSHIP'
    elif line_cat == TEXT_CATEGORY and line_score > TEXT_CUTOFF:
        line_class = 'TEXT'
    elif date is not None:
        line_class = 'DATE'
        line_value = date

    return {**prediction_item, 'class': line_class, 'value': line_value, 'text': line_text_value}

def predict_line(line: str):
    """Predicts the category of a line using FastText only."""
    cleaned_line = preprocessor.preproc_one(line).replace("\n", " ").strip()

    if cleaned_line:
        prediction, probability = line_classifier.predict_one(cleaned_line, incl_probabilities=True).pop()
        prediction = int(prediction.replace('__label__', ''))
    else:
        prediction = 0
        probability = 0

    return {
        'category_id': prediction,
        'category_score': probability,
        'line': line,
        'clean_line': cleaned_line
    }
