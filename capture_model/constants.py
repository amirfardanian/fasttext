import re

TOTAL_CATEGORY = 99100
DISCOUNT_CATEGORY = 99200
MOMS_CATEGORY = 99500
CVR_CATEGORY = 99951
CHANGE_CATEGORY = 99400
MEMBERSHIP_CATEGORY = 99800
TEXT_CATEGORY = -1
PHONE_NUMBER_CATEGORY = -2
ADDRESS_CATEGORY = -3
UNKNOWN_CATEGORY = -4

TOTAL_CUTOFF = 0.7
DISCOUNT_CUTOFF = 0.5
MOMS_CUTOFF = 0.7
CVR_CUTOFF = 0.3
CHANGE_CUTOFF = 0.9
TEXT_CUTOFF = 0.6

DATE_FORMAT_REGEXES = [
    {
        'regexp': re.compile(r'(\d{4}[ ]?-[ ]?\d{2}[ ]?-[ ]?\d{2}\s+\d{2}[ ]?[:.][ ]?\d{2})'),
        'formats': ['%Y-%m-%d %H:%M', '%Y-%m-%d %H.%M']
    },
    {
        'regexp': re.compile(r'(\d{2}[ ]?[-./][ ]?\d{2}[ ]?[-./][ ]?\d{4}\s+\d{2}[ ]?[:.][ ]?\d{2})'),
        'formats': ['%d-%m-%Y %H:%M', '%d.%m.%Y %H:%M', '%d/%m/%Y %H:%M',
                    '%d-%m-%Y %H.%M', '%d.%m.%Y %H.%M', '%d/%m/%Y %H.%M']
    },
    {
        'regexp': re.compile(r'(\d{2}[ ]?[-./ ][ ]?\d{2}[ ]?[-./ ][ ]?\d{2}\s+\d{2}[ ]?[:.][ ]?\d{2})'),
        'formats': ['%d-%m-%y %H:%M', '%d.%m.%y %H:%M', '%d/%m/%y %H:%M', '%d %m %y %H:%M',
                    '%d-%m-%y %H.%M', '%d.%m.%y %H.%M', '%d/%m/%y %H.%M', '%d %m %y %H.%M']
    },
    {
        'regexp': re.compile(r'(\d{2}[ ]?[-./][ ]?\d{2}[ ]?[-./][ ]?\d{2}\s+\d{2}[ ]?[:.][ ]?\d{2})'),
        'formats': ['%y-%m-%d %H:%M', '%y.%m.%d %H:%M', '%y/%m/%d %H:%M',
                    '%y-%m-%d %H.%M', '%y.%m.%d %H.%M', '%y/%m/%d %H.%M']
    },
    {
        'regexp': re.compile(r'(\d{4}[ ]?[-./][ ]?\d{2}[ ]?[-./][ ]?\d{2})'),
        'formats': ['%Y-%m-%d', '%Y.%m.%d', '%Y/%m/%d']
    },
    {
        'regexp': re.compile(r'(\d{2}[ ]?[-./ ][ ]?\d{2}[ ]?[-./ ][ ]?\d{4})'),
        'formats': ['%d.%m.%Y', '%d/%m/%Y', '%d-%m-%Y', '%d %m %Y']
    },
    {
        'regexp': re.compile(r'(\d{2}[ ]?[-./][ ]?\d{2}[ ]?[-./][ ]?\d{2})'),
        'formats': ['%y.%m.%d', '%y/%m/%d', '%y-%m-%d']
    },
    {
        'regexp': re.compile(r'(\d{2}[ ]?[-./][ ]?\d{2}[ ]?[-./][ ]?\d{2})'),
        'formats': ['%d-%m-%y', '%d.%m.%y', '%d/%m/%y']
    }
]

PRICE_REGEX = re.compile('(\d+[,.]\d{3}\s*[,.]\s*\d{2}(?:\D|$)|\d+\s*[,.]\s*\d{2}(?:\D|$)|\d+\s*[,.]\s*\d$)')
PRODUCT_NAME_REGEX = re.compile('(.+) \-{0,1}(\d+[,.]\d{3}\s*[,.]\s*\d{2}(?:\D|$)|\d+\s*[,.]\s*\d{2}(?:\D|$)|\d+\s*[,.]\s*\d$)')
CVR_REGEX = re.compile('\d{8}')