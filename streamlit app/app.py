import streamlit as st
import pandas as pd
import pickle
from sklearn.datasets import load_iris


# ---------------------------------------------------
# Load Iris dataset
# ---------------------------------------------------

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=[
        "Sepal Length",
        "Sepal Width",
        "Petal Length",
        "Petal Width"
    ]
)

df["Flower"] = iris.target_names[iris.target]


# ---------------------------------------------------
# Load trained model
# ---------------------------------------------------

with open("iris.pkl", "rb") as file:
    model = pickle.load(file)


# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

st.sidebar.title("🌸 Iris Flower App")

page = st.sidebar.radio(
    "Select Page",
    [
        "General Information",
        "Data Analysis",
        "Flower Prediction"
    ]
)


# ===================================================
# PAGE 1 - GENERAL INFORMATION
# ===================================================

if page == "General Information":

    st.title("🌸 Iris Flower Classification")

    st.write(
        "Welcome to the Iris Flower Machine Learning App."
    )

    st.write(
        "This application uses the Iris dataset to analyze "
        "flower measurements and predict the type of Iris flower."
    )

    st.header("About the Iris Dataset")

    st.write(
        "The Iris dataset contains measurements of three "
        "different types of Iris flowers."
    )

    st.subheader("Flower Types")

    st.write("🌸 Setosa")
    st.write("🌸 Versicolor")
    st.write("🌸 Virginica")

    st.subheader("Features")

    st.write("• Sepal Length")
    st.write("• Sepal Width")
    st.write("• Petal Length")
    st.write("• Petal Width")

    st.subheader("Machine Learning Model")

    st.write(
        "A Logistic Regression model is used to classify "
        "the Iris flower based on the four measurements."
    )


# ===================================================
# PAGE 2 - DATA ANALYSIS
# ===================================================

elif page == "Data Analysis":

    st.title("📊 Iris Data Analysis")

    st.write("Explore the Iris dataset.")

    # Dataset preview
    st.header("Dataset")

    st.dataframe(df)

    # Dataset information
    st.header("Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Rows",
            df.shape[0]
        )

    with col2:
        st.metric(
            "Columns",
            df.shape[1]
        )

    with col3:
        st.metric(
            "Flower Types",
            df["Flower"].nunique()
        )

    # Statistics
    st.header("Statistical Summary")

    st.dataframe(
        df.describe()
    )

    # Flower count
    st.header("Flower Count")

    flower_count = df["Flower"].value_counts()

    st.bar_chart(flower_count)

    # Feature selection
    st.header("Feature Analysis")

    selected_feature = st.selectbox(
        "Select a feature",
        [
            "Sepal Length",
            "Sepal Width",
            "Petal Length",
            "Petal Width"
        ]
    )

    st.write(
        "Average:",
        round(df[selected_feature].mean(), 2)
    )

    st.write(
        "Minimum:",
        round(df[selected_feature].min(), 2)
    )

    st.write(
        "Maximum:",
        round(df[selected_feature].max(), 2)
    )


# ===================================================
# PAGE 3 - FLOWER PREDICTION
# ===================================================

elif page == "Flower Prediction":

    st.title("🔮 Iris Flower Prediction")

    st.write(
        "Enter the flower measurements to predict the Iris type."
    )

    # Input fields

    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=5.1
    )

    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=3.5
    )

    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=1.4
    )

    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=0.2
    )


    # Prediction button

    if st.button("Predict Flower"):

        input_data = [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]]

        prediction = model.predict(input_data)

        flower_names = [
            "Setosa",
            "Versicolor",
            "Virginica"
        ]

        result = flower_names[prediction[0]]

        st.success(
            f"🌸 Predicted Flower: {result}"
        )