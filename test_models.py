import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("GREY MARKET DETECTION MODEL - TEST SUITE")
print("=" * 80)
print()

# Load models
print("📦 Loading models...")
try:
    lr_model = joblib.load("logistic_grey_market_model.pkl")
    xgb_model = joblib.load("grey_market_xgboost_model.pkl")
    print("✅ Models loaded successfully!")
except FileNotFoundError as e:
    print(f"❌ Error loading models: {e}")
    exit(1)

print()

# Test data with 20 different inputs
test_data = [
    {
        "Product_Title": "Apple iPhone 15 Pro Max 256GB",
        "Selling_Price": 130000,
        "MRP": 159900,
        "Discount_Pct": 18,
        "Review_Count": 2847,
        "Expected": 0,  # Legitimate - authorized seller, many reviews
        "Reason": "High review count, reasonable discount, premium product"
    },
    {
        "Product_Title": "Samsung Galaxy S24 Ultra",
        "Selling_Price": 85000,
        "MRP": 124999,
        "Discount_Pct": 32,
        "Review_Count": 1523,
        "Expected": 0,
        "Reason": "Popular product, legitimate pricing"
    },
    {
        "Product_Title": "Rolex Submariner Watch",
        "Selling_Price": 15000,
        "MRP": 850000,
        "Discount_Pct": 98,
        "Review_Count": 12,
        "Expected": 1,  # Grey Market - unrealistic discount
        "Reason": "98% discount on luxury watch - clear indicator of grey/fake product"
    },
    {
        "Product_Title": "Sony WH-1000XM5 Headphones",
        "Selling_Price": 19999,
        "MRP": 29990,
        "Discount_Pct": 33,
        "Review_Count": 3456,
        "Expected": 0,
        "Reason": "Reasonable discount, high authentic reviews"
    },
    {
        "Product_Title": "Canon EOS R5 Camera",
        "Selling_Price": 280000,
        "MRP": 350000,
        "Discount_Pct": 20,
        "Review_Count": 156,
        "Expected": 0,
        "Reason": "Premium camera, reasonable discount, moderate reviews"
    },
    {
        "Product_Title": "Designer Gucci Handbag",
        "Selling_Price": 5000,
        "MRP": 250000,
        "Discount_Pct": 98,
        "Review_Count": 8,
        "Expected": 1,
        "Reason": "Suspiciously low price for designer item, few reviews"
    },
    {
        "Product_Title": "Microsoft Surface Laptop Pro",
        "Selling_Price": 75000,
        "MRP": 99900,
        "Discount_Pct": 25,
        "Review_Count": 876,
        "Expected": 0,
        "Reason": "Standard laptop discount range"
    },
    {
        "Product_Title": "IPhone 14 Display High Quality",
        "Selling_Price": 8000,
        "MRP": 450000,
        "Discount_Pct": 98,
        "Review_Count": 15,
        "Expected": 1,
        "Reason": "Fake/grey market - extremely high discount, vague title"
    },
    {
        "Product_Title": "Nike Air Jordan 1 Retro High",
        "Selling_Price": 8500,
        "MRP": 15000,
        "Discount_Pct": 43,
        "Review_Count": 567,
        "Expected": 0,
        "Reason": "Reasonable sneaker pricing and discount"
    },
    {
        "Product_Title": "Omega Seamaster Watch Premium",
        "Selling_Price": 12000,
        "MRP": 600000,
        "Discount_Pct": 98,
        "Review_Count": 25,
        "Expected": 1,
        "Reason": "Luxury watch at 98% discount - clear counterfeit indicator"
    },
    {
        "Product_Title": "Dell XPS 15 Laptop i7",
        "Selling_Price": 95000,
        "MRP": 125000,
        "Discount_Pct": 24,
        "Review_Count": 1245,
        "Expected": 0,
        "Reason": "Legitimate laptop with standard discount"
    },
    {
        "Product_Title": "Louis Vuitton Speedy Bag",
        "Selling_Price": 3500,
        "MRP": 180000,
        "Discount_Pct": 98,
        "Review_Count": 5,
        "Expected": 1,
        "Reason": "Extremely high discount on luxury brand - grey market"
    },
    {
        "Product_Title": "LG OLED TV 55 inch",
        "Selling_Price": 85000,
        "MRP": 120000,
        "Discount_Pct": 29,
        "Review_Count": 432,
        "Expected": 0,
        "Reason": "Standard TV discount and review count"
    },
    {
        "Product_Title": "Prada Sunglasses",
        "Selling_Price": 2000,
        "MRP": 120000,
        "Discount_Pct": 98,
        "Review_Count": 10,
        "Expected": 1,
        "Reason": "Designer item at 98% discount"
    },
    {
        "Product_Title": "OnePlus 12 Smartphone",
        "Selling_Price": 42000,
        "MRP": 56000,
        "Discount_Pct": 25,
        "Review_Count": 2134,
        "Expected": 0,
        "Reason": "Popular phone with normal discount"
    },
    {
        "Product_Title": "Chanel No 5 Perfume Authentic",
        "Selling_Price": 1500,
        "MRP": 95000,
        "Discount_Pct": 98,
        "Review_Count": 20,
        "Expected": 1,
        "Reason": "95% discount on luxury perfume - suspicious"
    },
    {
        "Product_Title": "HP Pavilion Gaming Laptop",
        "Selling_Price": 55000,
        "MRP": 75000,
        "Discount_Pct": 27,
        "Review_Count": 789,
        "Expected": 0,
        "Reason": "Budget gaming laptop, legitimate pricing"
    },
    {
        "Product_Title": "Hermes Birkin Bag Leather",
        "Selling_Price": 8000,
        "MRP": 500000,
        "Discount_Pct": 98,
        "Review_Count": 3,
        "Expected": 1,
        "Reason": "Ultra-luxury item at extreme discount, very few reviews"
    },
    {
        "Product_Title": "Realme GT 6T Phone",
        "Selling_Price": 32000,
        "MRP": 45000,
        "Discount_Pct": 29,
        "Review_Count": 1654,
        "Expected": 0,
        "Reason": "Mid-range phone with normal pricing"
    },
    {
        "Product_Title": "Cartier Diamond Ring",
        "Selling_Price": 5000,
        "MRP": 300000,
        "Discount_Pct": 98,
        "Review_Count": 8,
        "Expected": 1,
        "Reason": "Jewelry with extreme discount - counterfeit indicator"
    }
]

# Create DataFrame
test_df = pd.DataFrame([{k: v for k, v in item.items() if k not in ["Expected", "Reason"]} for item in test_data])
expected_labels = np.array([item["Expected"] for item in test_data])

print(f"🧪 Testing {len(test_data)} different product entries...\n")

# Get predictions from both models
lr_predictions = lr_model.predict(test_df)
lr_probabilities = lr_model.predict_proba(test_df)[:, 1]

xgb_predictions = xgb_model.predict(test_df)
xgb_probabilities = xgb_model.predict_proba(test_df)[:, 1]

# Create results
results = []
for i, item in enumerate(test_data):
    results.append({
        "Index": i + 1,
        "Product": item["Product_Title"][:40],
        "Expected": "Grey Market" if expected_labels[i] == 1 else "Legitimate",
        "LR_Pred": "Grey Market" if lr_predictions[i] == 1 else "Legitimate",
        "LR_Conf": f"{lr_probabilities[i]*100:.1f}%",
        "XGB_Pred": "Grey Market" if xgb_predictions[i] == 1 else "Legitimate",
        "XGB_Conf": f"{xgb_probabilities[i]*100:.1f}%",
        "LR_Match": "✅" if lr_predictions[i] == expected_labels[i] else "❌",
        "XGB_Match": "✅" if xgb_predictions[i] == expected_labels[i] else "❌",
    })

# Display detailed results
print("=" * 150)
print(f"{'#':<3} {'Product Title':<40} {'Expected':<15} {'LR Prediction':<20} {'XGB Prediction':<20} {'Match':<15}")
print("=" * 150)

for i, item in enumerate(test_data):
    lr_match = "✅" if lr_predictions[i] == expected_labels[i] else "❌"
    xgb_match = "✅" if xgb_predictions[i] == expected_labels[i] else "❌"
    expected_label = "Grey Market" if expected_labels[i] == 1 else "Legitimate"
    lr_pred = "Grey Market" if lr_predictions[i] == 1 else "Legitimate"
    xgb_pred = "Grey Market" if xgb_predictions[i] == 1 else "Legitimate"
    
    print(f"{i+1:<3} {test_data[i]['Product_Title'][:40]:<40} {expected_label:<15} {lr_pred:<20} {xgb_pred:<20} LR:{lr_match} XGB:{xgb_match}")

print("=" * 150)
print()

# Detailed breakdown with confidence scores and reasoning
print("\n" + "=" * 80)
print("DETAILED ANALYSIS WITH CONFIDENCE SCORES")
print("=" * 80)
print()

for i, item in enumerate(test_data):
    expected_label = "Grey Market" if expected_labels[i] == 1 else "Legitimate"
    lr_pred = "Grey Market" if lr_predictions[i] == 1 else "Legitimate"
    xgb_pred = "Grey Market" if xgb_predictions[i] == 1 else "Legitimate"
    lr_match = "✅ CORRECT" if lr_predictions[i] == expected_labels[i] else "❌ INCORRECT"
    xgb_match = "✅ CORRECT" if xgb_predictions[i] == expected_labels[i] else "❌ INCORRECT"
    
    print(f"Test #{i+1}")
    print(f"├─ Product: {item['Product_Title']}")
    print(f"├─ Details: Price ₹{item['Selling_Price']} | MRP ₹{item['MRP']} | Discount {item['Discount_Pct']}% | Reviews {item['Review_Count']}")
    print(f"├─ Expected Label: {expected_label}")
    print(f"├─ Logistic Regression:")
    print(f"│  ├─ Prediction: {lr_pred} (Confidence: {lr_probabilities[i]*100:.2f}%)")
    print(f"│  └─ Status: {lr_match}")
    print(f"├─ XGBoost:")
    print(f"│  ├─ Prediction: {xgb_pred} (Confidence: {xgb_probabilities[i]*100:.2f}%)")
    print(f"│  └─ Status: {xgb_match}")
    print(f"└─ Reason: {item['Reason']}")
    print()

# Summary Statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print()

lr_accuracy = accuracy_score(expected_labels, lr_predictions)
xgb_accuracy = accuracy_score(expected_labels, xgb_predictions)

lr_precision = precision_score(expected_labels, lr_predictions, zero_division=0)
xgb_precision = precision_score(expected_labels, xgb_predictions, zero_division=0)

lr_recall = recall_score(expected_labels, lr_predictions, zero_division=0)
xgb_recall = recall_score(expected_labels, xgb_predictions, zero_division=0)

lr_f1 = f1_score(expected_labels, lr_predictions, zero_division=0)
xgb_f1 = f1_score(expected_labels, xgb_predictions, zero_division=0)

print(f"📊 Test Set Size: {len(test_data)} products")
print(f"   ├─ Expected Legitimate: {(expected_labels == 0).sum()}")
print(f"   └─ Expected Grey Market: {(expected_labels == 1).sum()}")
print()

print("🔹 LOGISTIC REGRESSION MODEL")
print(f"   ├─ Accuracy:  {lr_accuracy*100:.2f}% ({(lr_predictions == expected_labels).sum()}/{len(test_data)} correct)")
print(f"   ├─ Precision: {lr_precision*100:.2f}% (of predicted grey market, how many are actual)")
print(f"   ├─ Recall:    {lr_recall*100:.2f}% (of actual grey market, how many detected)")
print(f"   └─ F1-Score:  {lr_f1:.4f}")
print()

print("🔹 XGBOOST MODEL")
print(f"   ├─ Accuracy:  {xgb_accuracy*100:.2f}% ({(xgb_predictions == expected_labels).sum()}/{len(test_data)} correct)")
print(f"   ├─ Precision: {xgb_precision*100:.2f}% (of predicted grey market, how many are actual)")
print(f"   ├─ Recall:    {xgb_recall*100:.2f}% (of actual grey market, how many detected)")
print(f"   └─ F1-Score:  {xgb_f1:.4f}")
print()

# Confusion Matrices
lr_cm = confusion_matrix(expected_labels, lr_predictions)
xgb_cm = confusion_matrix(expected_labels, xgb_predictions)

print("📈 CONFUSION MATRICES")
print()
print("Logistic Regression:")
print(f"   True Negatives:  {lr_cm[0,0]:<3} | False Positives: {lr_cm[0,1]:<3}")
print(f"   False Negatives: {lr_cm[1,0]:<3} | True Positives:  {lr_cm[1,1]:<3}")
print()
print("XGBoost:")
print(f"   True Negatives:  {xgb_cm[0,0]:<3} | False Positives: {xgb_cm[0,1]:<3}")
print(f"   False Negatives: {xgb_cm[1,0]:<3} | True Positives:  {xgb_cm[1,1]:<3}")
print()

# Best performing model
print("=" * 80)
if xgb_accuracy > lr_accuracy:
    print(f"🏆 WINNER: XGBoost Model (Accuracy: {xgb_accuracy*100:.2f}%)")
elif lr_accuracy > xgb_accuracy:
    print(f"🏆 WINNER: Logistic Regression Model (Accuracy: {lr_accuracy*100:.2f}%)")
else:
    print(f"🏆 TIE: Both models have equal accuracy ({lr_accuracy*100:.2f}%)")

print("=" * 80)
