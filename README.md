# Temporal Analysis of Pallidus MR1 Data

This project was conducted for [KravitzLab](https://kravitzlab.com/), focusing on the activity tracking data of subjects over time. The analysis involves identifying temporal patterns, testing hypotheses, and comparing day versus night activity levels.

### Project Dependencies
The project was developed using Python with libraries such as 
- `Pandas` for data manipulation.
- `scipy` for statistical analysis.
- `Plotly` for interactive visualizations
- `Streamlit` for building the web app. 

### Getting Started
Follow these instructions to set up your environment and run the app locally.

#### Prerequisites
- Python 3.7+
- pip

#### Setup
1. Clone the repository.
2. Create and activate a virtual environment.
    ```
   python -m venv venv
   source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```
3. Install the required packages.
    ```
    pip install -r requirements.txt
    ```
4. Run the Streamlit app
   ```
   streamlit run dashboard.py
   ```

### Usage
- Upload Data: Use the sidebar to upload your MR1 activity data in Excel format.
- Select Analysis Options: Choose the type of analysis you want to perform from the horizontal menu.
- View Results: Interactive charts and tables will be displayed based on your selections.
- Download Summary: Download the generated summaries as an Excel file.

---