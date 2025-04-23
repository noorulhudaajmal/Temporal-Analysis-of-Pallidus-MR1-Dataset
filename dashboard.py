# Import the necessary libraries
import streamlit as st
from streamlit_option_menu import option_menu as option_menu

from files import *

# Define the HTML template for the dashboard styling
temp = """
<div style="background-color:{};padding:0.5px;border-radius:5px;">
<h4 style="color:{};text-align:center;display:in-line">{}</h6>
</div>
"""

# ------------------------------------------PAGE CONFIG-------------------------------------

st.set_page_config(page_title="Data Analysis", layout="wide")

# ------------------------------------------------------------------------------------------

# Style to add some margin to the block
st.markdown("""
  <style>
    .block-container {
      margin-top: 20px;
      padding-top:25px;
    }
  </style>
""", unsafe_allow_html=True)


# Function to pre-process the uploaded file
def pre_process(data):
    data.rename(columns={'Date & Time': 'DateTime'}, inplace=True)
    data["DateTime"] = pd.to_datetime(data["DateTime"])

    data = data.dropna(subset=['DateTime'])
    unnamed_cols = [col for col in data.columns if 'Unnamed' in col or 'Notes' in col]
    data = data.drop(columns=unnamed_cols, axis=1)

    # Set 'Date & Time' as the index
    data = data.set_index('DateTime', drop=False)

    return data


with st.sidebar:
    uploaded_file = st.file_uploader(
        label="Choose file:", type=["xls", "xlsx"])

    DAY_RANGE = st.slider(min_value=0, max_value=24,
                          label="Choose day hours range", value=[6, 18])

df = pd.DataFrame()
devices = []
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file, skiprows=3, header=1)
    df = pre_process(df)
    devices = list(df.columns[1:])

    menu = option_menu(menu_title=None,
                       options=["Raw Data", "Grouped Data",
                                "Stat Tests", "Summary"],
                       default_index=0,
                       icons=["barchart", "traces"],
                       orientation="horizontal")

    if menu == "Raw Data":
        hourly_average = df.resample('H').mean()
        st.plotly_chart(get_line_plot(hourly_average, day_range=DAY_RANGE), use_container_width=True)

    if menu == "Grouped Data":
        grouped_data = df.copy()
        G01 = st.sidebar.text_input(label="Group 1 name:", value="Group A", key="g1_name")
        G02 = st.sidebar.text_input(label="Group 1 name:", value="Group B", key="g2_name")

        col1, gap, col2 = st.columns((8, 1, 8))

        with col1:
            G1 = st.multiselect(label=f"Select Devices for {G01}",
                                default=[x[-3:] for x in devices], options=[x[-3:] for x in devices],
                                key="group1")
            user_date_input = st.slider(
                "Select Date Range:",
                min_value=df["DateTime"].min().to_pydatetime(),
                max_value=df["DateTime"].max().to_pydatetime(),
                value=(
                    df["DateTime"].min().to_pydatetime(),
                    df["DateTime"].max().to_pydatetime(),
                ), key=f"date_g1"
            )
            start_date1, end_date1 = tuple(map(pd.to_datetime, user_date_input))
            with col2:
                G2 = st.multiselect(label=f"Select Devices for {G02}",
                                    default=[x[-3:] for x in devices], options=[x[-3:] for x in devices],
                                    key="group2")
                user_date_input = st.slider(
                    "Select Date Range:",
                    min_value=df["DateTime"].min().to_pydatetime(),
                    max_value=df["DateTime"].max().to_pydatetime(),
                    value=(
                        df["DateTime"].min().to_pydatetime(),
                        df["DateTime"].max().to_pydatetime(),
                    ), key=f"date_g2"
                )
                start_date2, end_date2 = tuple(map(pd.to_datetime, user_date_input))

        g1_frame = grouped_data.loc[
            grouped_data["DateTime"].between(start_date1, end_date1.replace(hour=23, minute=59, second=59))]
        g2_frame = grouped_data.loc[
            grouped_data["DateTime"].between(start_date2, end_date2.replace(hour=23, minute=59, second=59))]

        st.plotly_chart(get_grouped_figure_test(g1_frame, g2_frame, G1, G2, G01, G02, day_range=DAY_RANGE),
                        use_container_width=True)

    if menu == "Stat Tests":
        G01 = st.sidebar.text_input(label="Group 1 name:", value="Group A", key="g1_name")
        G02 = st.sidebar.text_input(label="Group 1 name:", value="Group B", key="g2_name")

        col1, gap, col2 = st.columns((8, 1, 8))
        with col1:
            G1 = st.multiselect(label=f"Select Devices for {G01}",
                                default=[x[-3:] for x in devices], options=[x[-3:] for x in devices],
                                key="group1")

        with col2:
            G2 = st.multiselect(label=f"Select Devices for {G02}",
                                default=[x[-3:] for x in devices], options=[x[-3:] for x in devices],
                                    key="group2")

        st.plotly_chart(t_stats_bar_plots(G1, G2, df,
                                          names=[G01, G02],
                                          day_range=DAY_RANGE),
                        use_container_width=True)

    if menu == "Summary":
        G01 = st.sidebar.text_input(label="Group 1 name:", value="Group A", key="g1_name")
        G02 = st.sidebar.text_input(label="Group 1 name:", value="Group B", key="g2_name")

        col1, gap, col2 = st.columns((8, 1, 8))
        with col1:
            G1 = st.multiselect(label=f"Select Devices for {G01}",
                                default=[x[-3:] for x in devices], options=[x[-3:] for x in devices],
                                key="group1")

        with col2:
            G2 = st.multiselect(label=f"Select Devices for {G02}",
                                default=[x[-3:] for x in devices], options=[x[-3:] for x in devices],
                                key="group2")

        st.write("# ")

        summary1, summary2 = create_summary(df, G1, G2, G01, G02, DAY_RANGE)
        c1, c2 = st.columns((2, 1))
        with c1:
            styler = summary1.style.hide(axis="index")
            st.write(styler.to_html(), unsafe_allow_html=True)
        with c2:
            st.write("### ")
            styler = summary2.style
            st.write(styler.to_html(), unsafe_allow_html=True)

        st.write("### ")

        st.download_button(label='📥 Download Current Result',
                           data=to_excel(summary1, summary2),
                           file_name="summary.xlsx")

else:
    st.info("Upload Data!!!")

# ------------------------------------------------------------------------------------------
