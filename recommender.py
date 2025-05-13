import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(page_title="AI Product Recommender", layout="wide", initial_sidebar_state="collapsed")

# Navigation state using session_state
if "page" not in st.session_state:
    st.session_state.page = "explainability"

with st.sidebar:
    st.title("🛒 AI Recommender")
    
    # Capture clicks and trigger sidebar collapse
    if st.button("🛍️ Product Catalog"):
        st.session_state.page = "catalog"
        st.session_state.collapse_sidebar = True

    if st.button("🧠 Prediction + Explainability"):
        st.session_state.page = "explainability"
        st.session_state.collapse_sidebar = True
        
    st.markdown("---")
    st.markdown("## 💡 Project Overview")
    st.markdown("Predicts whether a customer will purchase a product and explains **why** using SHAP.")
    st.markdown("🔗 [GitHub Repo](https://github.com/MAX2YT/Explainable-ai.git)")
    

# Sample product data (use real image URLs)
product_list = [
    {"name": "Budget Smartphone", "price": 15000, "category": "Electronics", "desc": "Affordable smartphone with all basic features.", "image": "product/budget-phone.jpg"},
    {"name": "Premium Smartphone", "price": 50000, "category": "Electronics", "desc": "Top-tier smartphone with a powerful camera and performance.", "image": "product/premium-phone.jpg"},
    {"name": "Smart Watch", "price": 3000, "category": "Electronics", "desc": "Keep track of your fitness and health on the go.", "image": "product/smartwatch.jpg"},
    {"name": "College Laptop", "price": 30000, "category": "Electronics", "desc": "Budget-friendly laptop for students and everyday use.", "image": "product/college-laptop.jpg"},
    {"name": "Gaming Laptop", "price": 70000, "category": "Electronics", "desc": "High-performance laptop for gamers.", "image": "product/gaming-laptop.jpg"},
    {"name": "Laptop Bag", "price": 1000, "category": "Accessories", "desc": "Stylish and durable laptop bag.", "image": "product/laptop-bag.jpg"},
    {"name": "Men's Wear", "price": 2000, "category": "Fashion", "desc": "Comfortable and stylish clothes for men.", "image": "product/mens-wear.jpg"},
    {"name": "55-inch QLED TV", "price": 30000, "category": "Electronics", "desc": "Stunning visuals and immersive sound.", "image": "product/tv.jpg"},
    {"name": "PS5 Console", "price": 50000, "category": "Electronics", "desc": "Next-gen gaming console for incredible performance.", "image": "product/ps5.jpg"},
    {"name": "Women's Fashion Dress", "price": 2500, "category": "Fashion", "desc": "Elegant dress perfect for casual and formal occasions.", "image": "product/women-wear.jpg"},
    {"name": "Makeup Kit", "price": 3000, "category": "Beauty", "desc": "Complete makeup kit with high-quality products.", "image": "product/makeup.jpg"},
    {"name": "Handbag", "price": 3500, "category": "Accessories", "desc": "Stylish handbag for all occasions.", "image": "product/hand-bag.jpg"},
]

# Product recommendation function
def recommend_products(age, location, total_spent, gender):
    recommended = []
    if gender == "Female":
        if total_spent > 10000:
            recommended += ["Premium Smartphone", "Gaming Laptop", "PS5 Console", "Makeup Kit", "Handbag"]
        elif age > 25 and location == 0:
            recommended += ["College Laptop", "Smart Watch", "Women's Fashion Dress"]
        else:
            recommended += ["Smart Watch", "Laptop Bag", "Women's Fashion Dress"]
    else:
        if total_spent > 10000:
            recommended += ["Gaming Laptop", "PS5 Console"]
        elif age > 25 and location == 0:
            recommended += ["Premium Smartphone", "College Laptop"]
        else:
            recommended += ["Laptop Bag", "Smart Watch"]
    return recommended

# Synthetic data + cached model training
@st.cache_resource
def train_model():
    np.random.seed(42)
    df = pd.DataFrame({
        'Age': np.random.randint(18, 60, 100),
        'Location': np.random.choice(['Urban', 'Suburban', 'Rural'], 100),
        'TotalSpent': np.random.uniform(50, 100000, 100),
        'NumSiteVisits': np.random.randint(1, 20, 100),
        'ClickedAd': np.random.choice([0, 1], 100),
        'Gender': np.random.choice(['Male', 'Female'], 100),
    })
    df['Location'] = df['Location'].map({'Urban': 0, 'Suburban': 1, 'Rural': 2})
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    df['WillBuy'] = ((df['TotalSpent'] > 500) & (df['ClickedAd'] == 0)).astype(int)
    X = df.drop('WillBuy', axis=1)
    y = df['WillBuy']
    model = RandomForestClassifier()
    model.fit(X, y)
    return model, df

model, data = train_model()

# Product Catalog
if st.session_state.page == "catalog":
    st.title("🛍️ Featured Products")
    for product in product_list:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(product["image"], width=300)
        with col2:
            st.subheader(product["name"])
            st.write(product["desc"])
            st.markdown(f"**Price:** ₹{product['price']}")
        st.markdown("---")

# Prediction + Explainability
elif st.session_state.page == "explainability":
    st.title("🧠 Will the Customer Buy? + SHAP Explainability")

    age = st.slider("Age", 18, 60, 30)
    location = st.selectbox("Location", ['Urban', 'Suburban', 'Rural'])
    total_spent = st.slider("Total Amount Spent (₹)", 50.0, 100000.0, 200.0)
    site_visits = st.slider("Number of Site Visits", 1, 20, 5)
    clicked_ad = st.selectbox("Clicked on Ad?", [0, 1])
    gender = st.selectbox("Gender", ['Male', 'Female'])

    location_encoded = {'Urban': 0, 'Suburban': 1, 'Rural': 2}[location]
    gender_encoded = {'Male': 0, 'Female': 1}[gender]

    input_data = pd.DataFrame([[age, location_encoded, total_spent, site_visits, clicked_ad, gender_encoded]],
                              columns=['Age', 'Location', 'TotalSpent', 'NumSiteVisits', 'ClickedAd', 'Gender'])

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    st.subheader("🔍 Prediction:")
    st.write("✅ Will Buy" if prediction == 0 else "❌ Will Not Buy")
    st.write(f"Probability of Buying: **{proba:.2f}**")

    st.subheader("📊 Why did the model predict this?")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    if isinstance(shap_values, list):
        shap_to_use = shap_values[1]
    else:
        shap_to_use = shap_values

    # ✅ Fix applied here
    if shap_to_use.ndim == 3:
        shap_to_use = shap_to_use[:, :, 0]

    shap_df = pd.DataFrame(shap_to_use, columns=input_data.columns)

    fig, ax = plt.subplots(figsize=(8, 4))
    shap_df.iloc[0].sort_values(ascending=False).plot(kind='barh', ax=ax, color='skyblue')
    ax.set_title("Feature Impact on Prediction", fontsize=14)
    ax.set_xlabel("SHAP Value")
    st.pyplot(fig)

    with st.expander("ℹ️ What does this chart mean?"):
        st.markdown("""
        - **Positive SHAP Value**: Feature supports prediction of buying.
        - **Negative SHAP Value**: Feature pushes against buying.
        """)

    st.subheader("🎁 Recommended Products")
    recommended = recommend_products(age, location_encoded, total_spent, gender)
    for name in recommended:
        product = next((p for p in product_list if p["name"] == name), None)
        if product:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(product["image"], width=100)
            with col2:
                st.markdown(f"**{product['name']}**")
                st.write(product["desc"])
                st.markdown(f"**Price:** ₹{product['price']}")
            st.markdown("---")

