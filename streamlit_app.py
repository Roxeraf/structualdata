import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Function to load data from the uploaded file
def load_data(file):
    if file.type == 'text/csv':
        return pd.read_csv(file)
    elif file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
        return pd.read_excel(file)
    else:
        st.error('Unsupported file type.')
        return None

# Streamlit app
st.title('Structural Data Analysis')

# File upload
uploaded_file = st.file_uploader('Upload a CSV or Excel file', type=['csv', 'xlsx'])

if uploaded_file is not None:
    data = load_data(uploaded_file)
    if data is not None:
        # Display data preview
        st.subheader('Data Preview')
        st.write(data.head())

        # Display summary statistics
        st.subheader('Summary Statistics')
        st.write(data.describe())

        # Data Visualization
        st.subheader('Data Visualization')
        column_selection = st.multiselect('Select columns to visualize', data.columns)

        if column_selection:
            st.write('Histograms')
            for col in column_selection:
                st.write(f'Histogram for {col}')
                fig, ax = plt.subplots()
                sns.histplot(data[col], ax=ax, kde=True)
                st.pyplot(fig)

            st.write('Box Plots')
            for col in column_selection:
                st.write(f'Box Plot for {col}')
                fig, ax = plt.subplots()
                sns.boxplot(data=data[col], ax=ax)
                st.pyplot(fig)

            if len(column_selection) >= 2:
                st.write('Scatter Plot')
                x_col = st.selectbox('Select X-axis', column_selection, key='x_col')
                y_col = st.selectbox('Select Y-axis', column_selection, key='y_col')
                st.write(f'Scatter Plot between {x_col} and {y_col}')
                fig, ax = plt.subplots()
                sns.scatterplot(data=data, x=x_col, y=y_col, ax=ax)
                st.pyplot(fig)

        # Custom Analysis
        st.subheader('Custom Analysis')
        st.write('Filter and group data for custom analysis')

        # Example: Filter data
        filter_column = st.selectbox('Select column to filter', data.columns)
        filter_value = st.text_input('Enter value to filter')
        if filter_value:
            filtered_data = data[data[filter_column].astype(str) == filter_value]
            st.write(filtered_data)

        # Example: Group data
        group_column = st.selectbox('Select column to group by', data.columns)
        if group_column:
            grouped_data = data.groupby(group_column).mean()
            st.write(grouped_data)

# Run the app with: streamlit run app.py


