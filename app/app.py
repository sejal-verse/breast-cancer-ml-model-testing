import joblib
import pandas as pd
import streamlit as st
from pathlib import Path


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="PhishGuard",
    page_icon="🛡️",
    layout="wide"
)


# --------------------------------------------------
# Project Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "phishing_model.pkl"
FEATURE_PATH = PROJECT_ROOT / "models" / "feature_columns.pkl"


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURE_PATH)

    return model, features


model, feature_columns = load_model()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.title("🛡️ PhishGuard")

    st.markdown("### About")

    st.write(
        "PhishGuard is a machine-learning application "
        "for detecting patterns associated with phishing websites."
    )

    st.divider()

    st.markdown("### 📊 Model Information")

    st.write(f"**Model:** Decision Tree")
    st.write(f"**Features:** {len(feature_columns)}")
    st.write("**Task:** Binary Classification")

    st.divider()

    st.markdown("### 📚 Dataset")

    st.write("**Dataset:** UCI Phishing Websites")
    st.write("**Instances:** 11,055")
    st.write("**Features:** 30")

    st.divider()

    st.markdown("### 🔢 Feature Encoding")

    st.write("**-1** → Phishing-related")
    st.write("**0** → Neutral / uncertain")
    st.write("**1** → Legitimate-related")

    st.divider()

    st.caption(
        "This application demonstrates a trained ML model. "
        "It does not directly visit, scan, or inspect live websites."
    )
    
    
# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🛡️ PhishGuard")

st.subheader("Phishing Website Detection using Machine Learning")

st.write(
    "PhishGuard uses a trained machine-learning model to classify "
    "website characteristics as either phishing or legitimate."
)

st.divider()


# --------------------------------------------------
# About Section
# --------------------------------------------------

with st.expander("ℹ️ About this project"):

    st.write(
        "This project investigates how machine-learning models can "
        "classify phishing websites using website-related features."
    )

    st.write(
        "The model was trained using the UCI Phishing Websites dataset."
    )

    st.write(
        "The current version is a model demonstration. "
        "It does not directly scan or visit a website."
    )


# --------------------------------------------------
# Feature Encoding Explanation
# --------------------------------------------------

st.header("🔎 Website Characteristics")

st.info(
    """
The dataset encodes each website characteristic using values such as
**-1, 0, and 1**. The exact meaning depends on the individual feature.

For example, a feature may use these values to represent different
conditions or categories. The model learns how these encoded
characteristics relate to phishing and legitimate websites.

The target prediction itself is:

- **-1 → Phishing**
- **1 → Legitimate**
"""
)

#------------------------------------------
# Feature Descriptions
# --------------------------------------------------

feature_descriptions = {
    "having_IP_Address": (
        "IP Address in URL",
        "Checks whether the website URL uses an IP address instead of a domain name."
    ),

    "URL_Length": (
        "URL Length",
        "Indicates whether the URL has an unusually long length."
    ),

    "Shortining_Service": (
        "URL Shortening Service",
        "Checks whether the URL uses a URL-shortening service."
    ),

    "having_At_Symbol": (
        "@ Symbol in URL",
        "Checks whether the URL contains an @ symbol."
    ),

    "double_slash_redirecting": (
        "Double Slash Redirect",
        "Checks for suspicious use of double slashes in the URL."
    ),

    "Prefix_Suffix": (
        "Prefix or Suffix",
        "Checks for a hyphen in the domain name."
    ),

    "having_Sub_Domain": (
        "Subdomain",
        "Indicates the number of subdomains used by the website."
    ),

    "SSLfinal_State": (
        "SSL Certificate",
        "Represents characteristics related to the website's SSL certificate."
    ),

    "Domain_registeration_length": (
        "Domain Registration Length",
        "Represents the registration period of the domain."
    ),

    "Favicon": (
        "Favicon",
        "Checks characteristics of the website's favicon."
    ),

    "port": (
        "Port",
        "Checks whether the website uses a potentially suspicious port."
    ),

    "HTTPS_token": (
        "HTTPS Token",
        "Checks characteristics of HTTPS usage in the domain."
    ),

    "Request_URL": (
        "Request URL",
        "Represents how external objects are requested by the webpage."
    ),

    "URL_of_Anchor": (
        "Anchor URL",
        "Represents characteristics of links pointing to other locations."
    ),

    "Links_in_tags": (
        "Links in HTML Tags",
        "Represents links contained within webpage tags."
    ),

    "SFH": (
        "Server Form Handler",
        "Represents the behavior of the form submission destination."
    ),

    "Submitting_to_email": (
        "Submitting to Email",
        "Checks whether form information is submitted through email."
    ),

    "Abnormal_URL": (
        "Abnormal URL",
        "Checks whether the URL has characteristics associated with abnormal URLs."
    ),

    "Redirect": (
        "Redirect",
        "Represents the number of redirects associated with the webpage."
    ),

    "on_mouseover": (
        "Mouseover Behavior",
        "Checks whether mouseover events modify or hide webpage behavior."
    ),

    "RightClick": (
        "Right-Click Behavior",
        "Checks whether right-click functionality is disabled."
    ),

    "popUpWindow": (
        "Popup Window",
        "Represents popup-window behavior."
    ),

    "Iframe": (
        "Iframe",
        "Checks whether the webpage uses iframe elements."
    ),

    "age_of_domain": (
        "Domain Age",
        "Represents the age of the website's domain."
    ),

    "DNSRecord": (
        "DNS Record",
        "Represents whether a DNS record exists for the domain."
    ),

    "web_traffic": (
        "Web Traffic",
        "Represents the website's traffic characteristics."
    ),

    "Page_Rank": (
        "Page Rank",
        "Represents the website's ranking-related characteristics."
    ),

    "Google_Index": (
        "Google Index",
        "Represents whether the website is indexed by Google."
    ),

    "Links_pointing_to_page": (
        "Links Pointing to Page",
        "Represents the number of links pointing toward the webpage."
    ),

    "Statistical_report": (
        "Statistical Report",
        "Represents characteristics identified through statistical analysis."
    )
}

# --------------------------------------------------
# Website Characteristics
# --------------------------------------------------

st.header("🔎 Website Characteristics")

st.write(
    "Enter the characteristics of the website. "
    "Each feature uses the encoding defined by the dataset."
)


# --------------------------------------------------
# Feature Groups
# --------------------------------------------------

feature_groups = {

    "🌐 URL & Address": [
        "having_IP_Address",
        "URL_Length",
        "Shortining_Service",
        "having_At_Symbol",
        "double_slash_redirecting",
        "Prefix_Suffix",
        "having_Sub_Domain",
    ],

    "🔗 Links & HTML": [
        "SSLfinal_State",
        "Domain_registeration_length",
        "Favicon",
        "port",
        "HTTPS_token",
        "Request_URL",
        "URL_of_Anchor",
        "Links_in_tags",
        "SFH",
        "Submitting_to_email",
        "Abnormal_URL",
    ],

    "🔐 Security & Identity": [
        "Redirect",
        "on_mouseover",
        "RightClick",
        "popUpWidnow",
        "Iframe",
    ],

    "📊 Website Behavior & Statistics": [
        "age_of_domain",
        "DNSRecord",
        "web_traffic",
        "Page_Rank",
        "Google_Index",
        "Links_pointing_to_page",
        "Statistical_report",
    ],
}


# --------------------------------------------------
# Collect User Inputs
# --------------------------------------------------

input_values = {}


for group_name, features in feature_groups.items():

    with st.expander(group_name, expanded=False):

        for feature in features:

            display_name, description = feature_descriptions.get(
                feature,
                (
                    feature,
                    "Dataset feature used by the machine-learning model."
                )
            )

            st.markdown(f"**{display_name}**")

            st.caption(description)

            input_values[feature] = st.selectbox(
                "Feature value",
                options=[-1, 0, 1],
                index=1,
                key=f"input_{feature}",
                label_visibility="collapsed"
            )


st.divider()


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button(
    "🔍 Analyze Website Characteristics",
    use_container_width=True
):

    input_data = pd.DataFrame(
        [input_values],
        columns=feature_columns
    )

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    class_probabilities = dict(
        zip(model.classes_, probabilities)
    )

    phishing_probability = class_probabilities.get(-1, 0)
    legitimate_probability = class_probabilities.get(1, 0)

    st.divider()

    st.header("📊 Analysis Result")

    # --------------------------------------------------
    # Prediction Card
    # --------------------------------------------------

    if prediction == -1:

        st.error(
            "⚠️ Potential Phishing Website"
        )

        st.write(
            "The model classified the provided website "
            "characteristics as **phishing**."
        )

    else:

        st.success(
            "✅ Likely Legitimate Website"
        )

        st.write(
            "The model classified the provided website "
            "characteristics as **legitimate**."
        )


    # --------------------------------------------------
    # Probability Metrics
    # --------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Phishing Probability",
            f"{phishing_probability:.2%}"
        )

    with col2:

        st.metric(
            "Legitimate Probability",
            f"{legitimate_probability:.2%}"
        )


    # --------------------------------------------------
    # Probability Visualization
    # --------------------------------------------------

    st.subheader("Model Probability")

    st.progress(
        float(phishing_probability),
        text=f"Phishing: {phishing_probability:.2%}"
    )

    st.progress(
        float(legitimate_probability),
        text=f"Legitimate: {legitimate_probability:.2%}"
    )


    # --------------------------------------------------
    # Interpretation
    # --------------------------------------------------

    st.subheader("🧠 Interpretation")

    if prediction == -1:

        st.warning(
            "The model found a pattern in the supplied feature values "
            "that is associated with phishing websites in the training data."
        )

    else:

        st.info(
            "The model found a pattern in the supplied feature values "
            "that is associated with legitimate websites in the training data."
        )


    st.caption(
        "Important: This is a machine-learning prediction based on "
        "the supplied dataset features. It is not a guarantee that "
        "a real-world website is safe or malicious."
    )
    

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "PhishGuard | Machine Learning Project | "
    "UCI Phishing Websites Dataset"
)

# --------------------------------------------------
# Model Performance
# --------------------------------------------------

st.divider()

st.header("📈 Model Performance")

st.write(
    "Final evaluation results for the baseline and tuned "
    "Decision Tree models."
)


# --------------------------------------------------
# File Paths
# --------------------------------------------------

RESULTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "final_results.csv"
)

CONFUSION_MATRIX_PATH = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "final_confusion_matrix.png"
)


# --------------------------------------------------
# Load Results
# --------------------------------------------------

if RESULTS_PATH.exists():

    results = pd.read_csv(
        RESULTS_PATH
    )

    # Convert metric columns to percentages
    display_results = results.copy()

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]

    for column in metric_columns:

        display_results[column] = (
            display_results[column] * 100
        ).round(2)


    # --------------------------------------------------
    # Display Comparison Table
    # --------------------------------------------------

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True
    )


else:

    st.warning(
        "Final evaluation results were not found."
    )


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

if CONFUSION_MATRIX_PATH.exists():

    st.subheader(
        "Confusion Matrix"
    )

    st.image(
        str(CONFUSION_MATRIX_PATH),
        caption="Improved Decision Tree - Confusion Matrix"
    )

else:

    st.warning(
        "Confusion matrix image was not found."
    )


# --------------------------------------------------
# Metric Explanation
# --------------------------------------------------

with st.expander(
    "📚 What do these metrics mean?"
):

    st.write(
        "**Accuracy:** Percentage of all predictions "
        "that were correct."
    )

    st.write(
        "**Precision:** Among websites predicted as "
        "phishing, how many were actually phishing."
    )

    st.write(
        "**Recall:** Among actual phishing websites, "
        "how many were correctly identified."
    )

    st.write(
        "**F1 Score:** A combined measure of precision "
        "and recall."
    )

    st.write(
        "**ROC-AUC:** Measures how well the model "
        "separates phishing and legitimate websites "
        "across classification thresholds."
    )

# --------------------------------------------------
# Feature Importance
# --------------------------------------------------

st.divider()

st.header("🔍 Feature Importance")

st.write(
    "Feature importance shows how much each website characteristic "
    "contributed to the Decision Tree's predictions."
)


FEATURE_IMPORTANCE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "feature_importance.csv"
)

FEATURE_IMPORTANCE_IMAGE = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "feature_importance.png"
)

# --------------------------------------------------
# Project Workflow
# --------------------------------------------------

st.divider()

st.header("🧭 Project Workflow")

st.write(
    "The project follows a systematic machine-learning workflow "
    "from dataset exploration to final model evaluation."
)

workflow_steps = [
    ("1️⃣ Dataset", "Loaded the UCI Phishing Websites dataset."),
    ("2️⃣ EDA", "Explored the dataset and examined feature patterns."),
    ("3️⃣ Preprocessing", "Prepared the dataset for machine-learning models."),
    ("4️⃣ Baseline Model", "Trained a Decision Tree as the baseline model."),
    ("5️⃣ Model Comparison", "Compared multiple machine-learning algorithms."),
    ("6️⃣ Error Analysis", "Examined incorrect phishing and legitimate predictions."),
    ("7️⃣ Robustness Testing", "Tested how model predictions changed under feature perturbation."),
    ("8️⃣ Feature Importance", "Analyzed which features influenced the Decision Tree."),
    ("9️⃣ Model Improvement", "Tested Decision Tree hyperparameters."),
    ("🔟 Final Evaluation", "Compared final model performance using multiple metrics."),
    ("1️⃣1️⃣ Interactive App", "Built this Streamlit interface for model demonstration."),
]


for step, description in workflow_steps:

    with st.expander(step):

        st.write(description)


# --------------------------------------------------
# Feature Importance Table
# --------------------------------------------------

if FEATURE_IMPORTANCE_PATH.exists():

    importance_df = pd.read_csv(
        FEATURE_IMPORTANCE_PATH
    )

    st.subheader("Most Important Features")

    st.dataframe(
        importance_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.warning(
        "Feature importance results were not found."
    )


# --------------------------------------------------
# Feature Importance Visualization
# --------------------------------------------------

if FEATURE_IMPORTANCE_IMAGE.exists():

    st.subheader("Feature Importance Visualization")

    st.image(
        str(FEATURE_IMPORTANCE_IMAGE),
        use_container_width=True
    )

else:

    st.warning(
        "Feature importance visualization was not found."
    )


# --------------------------------------------------
# Explanation
# --------------------------------------------------

with st.expander(
    "📚 What does feature importance mean?"
):

    st.write(
        "Feature importance indicates how strongly a feature "
        "contributed to the model's decision-making process."
    )

    st.write(
        "A higher importance value means that the feature had "
        "more influence on the trained Decision Tree compared "
        "with features having lower importance values."
    )

    st.info(
        "Feature importance does not mean that a feature alone "
        "causes a website to be phishing. It describes the "
        "feature's contribution within this trained model."
    )
    
   # --------------------------------------------------
# Project Workflow
# --------------------------------------------------

st.divider()

st.header("🧭 Project Workflow")

st.write(
    "The project follows a systematic machine-learning workflow "
    "from dataset exploration to final model evaluation."
)

workflow_steps = [
    ("1️⃣ Dataset", "Loaded the UCI Phishing Websites dataset."),
    ("2️⃣ EDA", "Explored the dataset and examined feature patterns."),
    ("3️⃣ Preprocessing", "Prepared the dataset for machine-learning models."),
    ("4️⃣ Baseline Model", "Trained a Decision Tree as the baseline model."),
    ("5️⃣ Model Comparison", "Compared multiple machine-learning algorithms."),
    ("6️⃣ Error Analysis", "Examined incorrect phishing and legitimate predictions."),
    ("7️⃣ Robustness Testing", "Tested prediction stability under feature perturbation."),
    ("8️⃣ Feature Importance", "Analyzed which features influenced the Decision Tree."),
    ("9️⃣ Model Improvement", "Tested Decision Tree hyperparameters."),
    ("🔟 Final Evaluation", "Compared model performance using multiple metrics."),
    ("1️⃣1️⃣ Interactive App", "Built this Streamlit interface for model demonstration."),
]

for step, description in workflow_steps:

    with st.expander(step):
        st.write(description)
        