import sys
sys.path.append('/home/ec2-user/environment/capture_model_old-master')

import os
from capture_model.modelling.fasttext_model import FasttextClassifierModel

# Define the model storage directory
package_data_dir = "/home/ec2-user/environment/capture_model_old-master/capture_model/package_data/"

# Load the trained FastText model
fasttext_model = FasttextClassifierModel.from_json(os.path.join(package_data_dir, "model_fasttext.json"))

# Example receipt lines to classify
test_lines = [
    # Prices
    "Cheese 12,50",
    "Water 1.5L 9,99",
    "Tomatoes 5,45",
    "Chicken Fillet 89,90",
    "Coca-Cola 2L 15,00",
    
    # Date Formats
    "Købsdato: 2024-03-11",
    "Udstedelsesdato: 11-03-2024",
    "Fakturadato: 11/03/24",
    
    # Total Amount
    "Total: 159,90 DKK",
    "Samlet beløb: 299,00",
    "TOTALT 450.50",
    
    # Discounts
    "Rabat 10% -15,00",
    "Kunde rabat -5,50",
    "Tilbudspris -30,00",
    
    # CVR (Company Registration Numbers)
    "CVR: 12345678",
    "Virksomhedsnummer: DK-87654321",
    
    # Membership-related
    "Medlemskort: 123456789",
    "Loyalitetsprogram aktiv",
    
    # Miscellaneous Receipt Elements
    "AFRUNDING                0,10-",
    "Alle dage kl. 08.00-20.00",
    "Bager Søndag 8",
    "MOMS 25% 50,00",
    "Byttegaranti gælder 30 dage"
]

# Classify each line using FastText only
print("\n✅ FastText Classification Results:")
for line in test_lines:
    fasttext_prediction = fasttext_model.predict_one(line, incl_probabilities=True)
    print(f"{line} → {fasttext_prediction}")
