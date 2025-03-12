import re
from datetime import datetime, timedelta
from capture_model.constants import PRICE_REGEX, CVR_REGEX, DATE_FORMAT_REGEXES

def find_prices(line):
    return PRICE_REGEX.findall(line)

def find_cvr(line):
    return CVR_REGEX.findall(line.replace(' ', ''))

def parse_date(line):
    for regexp in DATE_FORMAT_REGEXES:
        match = regexp['regexp'].search(line)
        if match:
            date_str = re.sub('\s+', ' ', match.group(1)).replace(' :', ':').replace(': ', ':').replace(' -', '-').replace('- ', '-')
            for format in regexp['formats']:
                try:
                    purchase_date = datetime.strptime(date_str, format)
                    if valid_date(purchase_date):
                        return purchase_date
                except ValueError:
                    continue
    return None

def pick_cvr(cvr_list):
    valid_cvrs = [cvr for cvr in cvr_list if valid_cvr(cvr)]
    return valid_cvrs[0] if valid_cvrs else None

def valid_date(date):
    now = datetime.now()
    two_years_ago = now.replace(year=now.year - 2)
    tomorrow = now + timedelta(days=1)
    return two_years_ago < date < tomorrow

def valid_cvr(cvr):
    weights = [2, 7, 6, 5, 4, 3, 2]
    cvr_digits = [int(digit) for digit in str(cvr)]
    if len(cvr_digits) != 8:
        return False
    seven_cvr_digits = cvr_digits[:-1]
    control_digit = cvr_digits[-1]
    mod_11 = sum(d * w for d, w in zip(seven_cvr_digits, weights)) % 11
    return (mod_11 == 0 and control_digit == 0) or (mod_11 != 1 and (11 - mod_11) == control_digit)

def valid_cvr(cvr):
    weights = [2, 7, 6, 5, 4, 3, 2]
    cvr_digits = [int(digit) for digit in str(cvr)]
    if len(cvr_digits) != 8:
        return False
    control_digit = cvr_digits.pop()
    cvr_weighted_sum = sum(cvr_digit * weight for cvr_digit, weight in zip(cvr_digits, weights))
    mod_11 = cvr_weighted_sum % 11
    return (mod_11 == 0 and control_digit == 0) or (mod_11 != 1 and (11 - mod_11) == control_digit)

def process_line(cleaned_line, prediction, probability, model_name):
    prices = find_prices(cleaned_line)
    cvr_numbers = find_cvr(cleaned_line)
    date = parse_date(cleaned_line)

    return {
        'line': cleaned_line,
        'prediction': prediction,
        'probability': probability,
        'model_used': model_name,
        'prices': prices,
        'cvr': pick_cvr(cvr_numbers),
        'date': date.strftime("%Y-%m-%d") if date else None
    }
