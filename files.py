import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import ttest_ind
from io import BytesIO
from plotly.subplots import make_subplots
import streamlit as st

import datetime

# Define the color palette for the dashboard
colors_list = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#457b9d", "#fb6f92", "#3a5a40", "#8ecae6",
               "#84a59d", "#370617", "#fb5607", "#ff006e", "#8338ec", "#70e000", "#ce4257", "#9c6644", "#ff7d00",
               "#233d4d", "#fff3b0", "#f94144", "#bb9457", "#f72585", "#bc96e6", "#e7ad99", "#ffd100", "#5e503f",
               "#ffee32", "#ffff3f", "#2885f0", "#fe6d73", "#d3f8e2", "#ffb700", "#be95c4", "#8ac926", "#b0228c",
               "#8cb369", "#e06d06", "#ffc53a", "#fe5d26", "#68b0ab", "#b9375e", "#8a5a44", "#fec3a6", "#6b4d57"]


def get_line_plot(df, day_range=[6, 18]):
    data = df.copy()
    if isinstance(data.DateTime.iloc[0], pd.Period):
        data.index = data.DateTime.apply(pd.Period.to_timestamp)
    else:
        data.index = pd.to_datetime(data.DateTime)
        data.DateTime = pd.to_datetime(data.DateTime)
    trace_data = []
    ind = 0
    for i in df.columns:
        if i == "DateTime":
            continue
        trace_data.append(go.Scatter(
            x=data.index, y=data[i], mode='lines',
            marker={'color': colors_list[ind]}, name=str(i)[-3:])
        )
        ind += 1
    fig = go.Figure(data=trace_data)
    data["hour"] = data.DateTime.dt.hour
    if (data.DateTime.dt.hour == 0).sum() != data.shape[0]:
        data["timeDay"] = data["hour"].apply(lambda x: "day" if x in list(
            range(day_range[0] + 1, day_range[1])) else "night")
        ind = data.timeDay.index
        vals = data.timeDay.values
        prev = 0
        times = []
        for i in range(1, data.timeDay.size - 1):
            if vals[i] != vals[prev]:
                if vals[prev] == "night":
                    times.append((ind[prev], ind[i - 1]))
                prev = i
        if vals[-1] == "night":
            times.append((ind[prev], ind[-1]))
        for time in times:
            fig.add_vrect(
                x0=time[0],
                x1=time[1],
                fillcolor="black",
                opacity=0.3,
                line_width=0,
            )
    fig.update_layout(hovermode="x unified", yaxis_title="Activity", xaxis_title="Time",
                      hoverlabel=dict(bgcolor="gray", font_color="black",
                                      font_size=16, font_family="Rockwell"),
                      xaxis=dict(rangeslider=dict(visible=True)),
                      height=600
                      )
    return fig


def get_grouped_data(data, columns, day_range=[6, 18]):
    all_devices = data.columns
    selected_columns = [x for x in all_devices if x[-3:] in columns]
    cols = ["DateTime"] + selected_columns
    df = data.copy()
    if len(columns) == 0:
        df["Mean"] = 0
    else:
        df = df[cols]
        df["Mean"] = df.iloc[:, 1:].mean(axis=1)
    df["Upper_Dev"] = df.Mean.apply(lambda x: x + df.Mean.std())
    df["Lower_Dev"] = df.Mean.apply(lambda x: x - df.Mean.std())

    if isinstance(df.DateTime.iloc[0], pd.Period):
        df.DateTime = df.DateTime.apply(pd.Period.to_timestamp)
    else:
        df.DateTime = pd.to_datetime(df.DateTime)

    df['hour'] = df['DateTime'].dt.hour
    df['day_night'] = df['hour'].apply(lambda x: "day" if x in range(
        day_range[0] + 1, day_range[1]) else "night")

    return df


def get_grouped_figure(df, G1, G2, g01, g02, day_range=[6, 18]):
    data = df.copy()
    trace_data = []

    if (len(G1) == 0) and (len(G2) == 0):
        trace_data.append(go.Scatter(x=list(df.index.values + 1), y=[]))
    elif (len(G2) != 0) or (len(G1) != 0):
        df1 = get_grouped_data(data, G1)
        df2 = get_grouped_data(data, G2)
        hover_text = None
        if sum(df1.Mean != 0) != 0:
            if isinstance(df1.DateTime.iloc[0], pd.Period):
                df1.DateTime = df1.DateTime.apply(pd.Period.to_timestamp)

            trace_data.append(go.Scatter(x=df1.DateTime, y=df1.Upper_Dev,
                                         fill=None,
                                         mode='lines', fillcolor="rgba(27, 67, 50, 0.4)",
                                         line=dict(color='rgba(255,255,255,0)'),
                                         showlegend=False, name="+ dev"
                                         ))
            trace_data.append(go.Scatter(
                x=df1.DateTime,
                y=df1.Lower_Dev,
                name="- dev",
                fill='tonexty',  # fill area between trace0 and trace1
                mode='lines', fillcolor="rgba(27, 67, 50, 0.4)", line=dict(color='rgba(255,255,255,0)'),
                showlegend=False))
            trace_data.append(
                go.Scatter(x=df1.DateTime, y=df1.Mean,
                           fill=None,
                           mode='lines', line=dict(color="rgba(27, 67, 50, 1)"),
                           name=g01)
            )

        if sum(df2.Mean != 0) != 0:
            if isinstance(df2.DateTime.iloc[0], pd.Period):
                df2.DateTime = df2.DateTime.apply(pd.Period.to_timestamp)
            trace_data.append(go.Scatter(x=df2.DateTime, y=df2.Upper_Dev,
                                         fill=None,
                                         mode='lines',
                                         line=dict(
                                             color='rgba(255,255,255,0)'),
                                         showlegend=False, name="+ dev"
                                         ))
            trace_data.append(go.Scatter(
                x=df2.DateTime,
                y=df2.Lower_Dev,
                fill='tonexty',  # fill area between trace0 and trace1
                mode='lines',
                fillcolor="rgba(164, 19, 60, 0.1)", line=dict(color='rgba(255,255,255,0)'),
                name="- dev",
                showlegend=False))
            trace_data.append(
                go.Scatter(x=df2.DateTime, y=df2.Mean,
                           fill=None,
                           mode='lines', line=dict(color="rgba(164, 19, 60, 1)"),
                           name=g02)
            )

    fig = go.Figure(data=trace_data)
    if isinstance(data.DateTime.iloc[0], pd.Period):
        data.DateTime = data.DateTime.apply(pd.Period.to_timestamp)
    elif ~isinstance(data.DateTime.iloc, pd.Timestamp):
        data.DateTime = data.DateTime.apply(pd.to_datetime)
    data["hour"] = data.DateTime.dt.hour
    if (data.DateTime.dt.hour == 0).sum() != data.shape[0]:
        data["timeDay"] = data["hour"].apply(lambda x: "day" if x in list(
            range(day_range[0] + 1, day_range[1])) else "night")
        ind = data.DateTime
        vals = data.timeDay.values
        prev = 0
        times = []
        for i in range(1, data.timeDay.shape[0] - 1):
            if vals[i] != vals[prev]:
                if vals[prev] == "night":
                    times.append((ind.iloc[prev], ind.iloc[i - 1]))
                prev = i
        if vals[-1] == "night":
            times.append((ind.iloc[prev], ind.iloc[-1]))

        for time in times:
            fig.add_vrect(
                x0=time[0],
                x1=time[1],
                fillcolor="black",
                opacity=0.3,
                line_width=0,
            )
    fig.update_layout(hovermode="x unified", yaxis_title="Value", xaxis_title="Time",
                      hoverlabel=dict(bgcolor="gray", font_color="black", font_size=16,
                                      font_family="Rockwell"),
                      xaxis=dict(rangeslider=dict(visible=True))
                      )

    return fig


def analysis_results(g1_df, g2_df, names, day_range=[6, 18]):
    g1_df["Group"] = names[0]
    g2_df["Group"] = names[1]
    if isinstance(g1_df.DateTime.iloc[0], pd.Period):
        g1_df.DateTime = g1_df.DateTime.apply(pd.Period.to_timestamp)
        g2_df.DateTime = g2_df.DateTime.apply(pd.Period.to_timestamp)
    else:
        g1_df.DateTime = pd.to_datetime(g1_df.DateTime)
        g2_df.DateTime = pd.to_datetime(g2_df.DateTime)

    g1_df['hour'] = g1_df['DateTime'].dt.hour
    g2_df['hour'] = g2_df['DateTime'].dt.hour
    g1_df['day_night'] = pd.cut(g1_df['hour'], bins=[0, day_range[0], day_range[1], 24],
                                labels=['Night', 'Day', 'Night'], ordered=False)
    g2_df['day_night'] = pd.cut(g2_df['hour'], bins=[0, day_range[0], day_range[1], 24],
                                labels=['Night', 'Day', 'Night'], ordered=False)

    all_day_data = pd.DataFrame(
        {"Means": [np.mean(g1_df.Mean), np.mean(g2_df.Mean)], "Cat": names})

    day_data = None
    night_data = None
    if (g1_df.day_night.isna().sum() == g1_df.shape[0]) or (g2_df.day_night.isna().sum() == g2_df.shape[0]):
        day_data = None
        night_data = None
    else:
        day_data = pd.DataFrame({"Means": [g1_df.groupby("day_night").mean()["Mean"]["Day"],
                                           g2_df.groupby("day_night").mean()["Mean"]["Day"]],
                                 "Cat": names})
        night_data = pd.DataFrame({"Means": [g1_df.groupby("day_night").mean()["Mean"]["Night"],
                                             g2_df.groupby("day_night").mean()["Mean"]["Night"]],
                                   "Cat": names})

    return all_day_data, day_data, night_data


def bar_comparison_plot(g1_df, g2_df, combined_df, TIME):
    if combined_df is None:
        return go.Figure(
            go.Scatter(x=[], y=[])
        )
    if TIME != "All":
        g1_df = g1_df[g1_df.day_night == TIME]
        g2_df = g2_df[g2_df.day_night == TIME]

    traces = [
        px.strip(g1_df, y=["Mean"], x="Group", stripmode='overlay',
                 color_discrete_sequence=["orange"]).data[0],
        px.strip(g2_df, y=["Mean"], x="Group", stripmode='overlay',
                 color_discrete_sequence=["red"]).data[0],
        go.Bar(x=combined_df.Cat, y=combined_df.Means, showlegend=False,
               textposition='outside',
               text=np.round(combined_df.Means, 2), textfont=dict(color='black', size=20),
               marker=dict(color=combined_df.Means,
                           colorscale="teal"))
    ]
    fig = go.Figure(traces)
    stat, p = ttest_ind(g1_df.Mean, g2_df.Mean)
    p_test_result = '<b>T-test Results:</b> <br>t-statistic: {:.3f} <br>p-value: {:.f}'.format(
        stat, p)
    fig.add_annotation(
        text=p_test_result, align='left', showarrow=False, xref="x", borderpad=10,
        font={"size": 25},
        yref="y", bgcolor="#d8e2dc", bordercolor="black", borderwidth=1
    )
    fig.update_layout(height=600, width=700,
                      xaxis_title="Groups", yaxis_title="Average", hovermode=False)
    return fig


def get_bar_plots(data, G1, G2, day_range, names=["Group1", "Group2"], pref="All"):
    g1_df = get_grouped_data(data, G1)
    g2_df = get_grouped_data(data, G2)
    all_day_data, day_data, night_data = analysis_results(
        g1_df, g2_df, names, day_range)
    if pref == "All":
        return bar_comparison_plot(g1_df, g2_df, all_day_data, pref)
    elif pref == "Day":
        return bar_comparison_plot(g1_df, g2_df, day_data, pref)
    elif pref == "Night":
        return bar_comparison_plot(g1_df, g2_df, night_data, pref)


def bar_comparison_plot_b(g1, g2, dt, frame, day_night, names=["Group1", "Group2"], day_range=[6, 18]):
    if frame is None:
        return go.Figure(
            go.Scatter(x=[], y=[])
        ), ""
    g1_df = get_grouped_data(dt, g1, day_range)
    g2_df = get_grouped_data(dt, g2, day_range)

    temp1 = g1_df.copy()
    temp2 = g2_df.copy()
    if day_night != "24/7":
        temp1 = temp1[temp1.day_night == day_night]
        temp2 = temp2[temp2.day_night == day_night]


    temp1['Mean'] = pd.to_numeric(temp1['Mean'], errors='coerce')
    temp2['Mean'] = pd.to_numeric(temp2['Mean'], errors='coerce')

    stat, p = ttest_ind(temp1.Mean, temp2.Mean)
    p_test_result = 't-statistic: {:.3f} <br>p-value: {:.10f}'.format(stat, p)

    df1 = temp1.copy()
    vals1 = [df1[x].mean() for x in df1.columns if x not in ["DateTime", "Mean", "Upper_Dev", "Lower_Dev", "hour", "day_night"]]
    grp1 = [names[0] for x in df1.columns if x not in ["DateTime", "Mean", "Upper_Dev", "Lower_Dev", "hour", "day_night"]]

    df2 = temp2.copy()
    vals2 = [df2[x].mean() for x in df2.columns if x not in ["DateTime", "Mean", "Upper_Dev", "Lower_Dev", "hour", "day_night"]]
    grp2 = [names[1] for x in df2.columns if x not in ["DateTime", "Mean", "Upper_Dev", "Lower_Dev", "hour", "day_night"]]

    i = 2
    traces = [
        go.Bar(x=frame.Cat, y=frame.Means, showlegend=False,
               marker=dict(color=frame.Means,
                           colorscale=colors_list)),

        px.strip(y=vals1, x=grp1, stripmode='overlay',
                 color_discrete_sequence=["gray"]).data[0],
        px.strip(y=vals2, x=grp2, stripmode='overlay',
                 color_discrete_sequence=["grey"]).data[0]
    ]
    fig = go.Figure(traces)
    fig.update_layout(height=800, width=700, xaxis_title="Groups",
                      yaxis_title="Average", hovermode=False)
    return fig, p_test_result


def analysis_results_b(g1, g2, frame, names, day_range=[6, 18]):
    g1_df = get_grouped_data(frame, g1, day_range)
    g2_df = get_grouped_data(frame, g2, day_range)
    g1_df["Group"] = names[0]
    g2_df["Group"] = names[1]
    if isinstance(g1_df.DateTime.iloc[0], pd.Period):
        g1_df.DateTime = g1_df.DateTime.apply(pd.Period.to_timestamp)
        g2_df.DateTime = g2_df.DateTime.apply(pd.Period.to_timestamp)
    else:
        g1_df.DateTime = pd.to_datetime(g1_df.DateTime)
        g2_df.DateTime = pd.to_datetime(g2_df.DateTime)

    g1_df['hour'] = g1_df['DateTime'].dt.hour
    g2_df['hour'] = g2_df['DateTime'].dt.hour
    g1_df['day_night'] = pd.cut(g1_df['hour'], bins=[0, day_range[0], day_range[1], 24],
                                labels=['night', 'day', 'night'], ordered=False)
    g2_df['day_night'] = pd.cut(g2_df['hour'], bins=[0, day_range[0], day_range[1], 24],
                                labels=['night', 'day', 'night'], ordered=False)

    all_day_data = pd.DataFrame(
        {"Means": [np.mean(g1_df.Mean), np.mean(g2_df.Mean)], "Cat": names})

    day_data = None
    night_data = None
    if (g1_df.day_night.isna().sum() == g1_df.shape[0]) or (g2_df.day_night.isna().sum() == g2_df.shape[0]):
        day_data = None
        night_data = None
    else:
        day_data = pd.DataFrame({"Means": [g1_df.groupby("day_night")["Mean"].mean().loc["day"],
                                           g2_df.groupby("day_night")["Mean"].mean().loc["day"]],
                                 "Cat": names})
        night_data = pd.DataFrame({"Means": [g1_df.groupby("day_night")["Mean"].mean().loc["night"],
                                             g2_df.groupby("day_night")["Mean"].mean().loc["night"]],
                                   "Cat": names})

    return all_day_data, day_data, night_data


def t_stats_bar_plots(g1, g2, frame, names, day_range):
    time_frames = analysis_results_b(g1, g2, frame, names, day_range)

    plot1, pt1 = bar_comparison_plot_b(g1, g2, frame, time_frames[0], "24/7", names, day_range)
    plot2, pt2 = bar_comparison_plot_b(g1, g2, frame, time_frames[1], "day", names, day_range)
    plot3, pt3 = bar_comparison_plot_b(g1, g2, frame, time_frames[2], "night", names, day_range)

    fig = make_subplots(rows=1, cols=3, shared_xaxes=True, vertical_spacing=0.17,
                        subplot_titles=(
                            f"<b>24/7</b><br>{pt1}", f"<b>Day Time</b><br>{pt2}", f"<b>Night Time</b><br>{pt3}"))

    x = 1
    for i in (plot1, plot2, plot3):
        for j in i.data:
            fig.append_trace(j, row=1, col=x)
        x += 1
    fig.update_layout(yaxis_title="Average")
    return fig


def create_summary(data, G1, G2, gn1, gn2, day_range):
    g1 = get_grouped_data(data, G1, day_range)
    g2 = get_grouped_data(data, G2, day_range)
    all_devices = data.columns
    G1 = [x for x in all_devices if x[-3:] in G1]
    G2 = [x for x in all_devices if x[-3:] in G2]
    xx = {'Devices': [],
          'Group': [],
          'Start Date': [],
          'End Date': [],
          'Day': [],
          'Night': [],
          'Total': []
          }
    for i in G1:
        xx["Devices"].append(i)
        xx["Group"].append(gn1)
        xx["Start Date"].append(g1.DateTime.min())
        xx["End Date"].append(g1.DateTime.max())
        if "day" in g1.day_night.value_counts():
            xx["Day"].append(g1[g1.day_night == "day"][i].mean())
        else:
            xx["Day"].append(0)
        if "night" in g1.day_night.value_counts():
            xx["Night"].append(g1[g1.day_night == "night"][i].mean())
        else:
            xx["Night"].append(0)
        xx["Total"].append(g1[i].mean())

    for i in G2:
        xx["Devices"].append(i)
        xx["Group"].append(gn2)
        xx["Start Date"].append(g2.DateTime.min())
        xx["End Date"].append(g2.DateTime.max())
        if ("day" in g2.day_night.value_counts()):
            xx["Day"].append(g2[g2.day_night == "day"][i].mean())
        else:
            xx["Day"].append(0)
        if ("night" in g2.day_night.value_counts()):
            xx["Night"].append(g2[g2.day_night == "night"][i].mean())
        else:
            xx["Night"].append(0)
        xx["Total"].append(g2[i].mean())

    main_frame = pd.DataFrame.from_dict(xx)

    g1['Mean'] = pd.to_numeric(g1['Mean'], errors='coerce')
    g2['Mean'] = pd.to_numeric(g2['Mean'], errors='coerce')

    stat, p = ttest_ind(g1.Mean, g2.Mean)
    night_stat, night_p = ttest_ind(
        g1[g1.day_night == "night"].Mean, g2[g2.day_night == "night"].Mean)
    day_stat, day_p = ttest_ind(
        g1[g1.day_night == "day"].Mean, g2[g2.day_night == "day"].Mean)

    side_panel = pd.DataFrame.from_dict({"t-stats": [day_stat, night_stat, stat], "p-value": [
        day_p, night_p, p]}, orient="index", columns=["Day", "Night", "Total"])

    return main_frame, side_panel


def to_excel(d1, d2):
    output = BytesIO()
    # Creating Excel Writer Object from Pandas
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    workbook = writer.book
    d1.to_excel(writer, sheet_name='Summary', startrow=0, startcol=0)
    d2.to_excel(writer, sheet_name='Summary',
                startrow=d1.shape[0] + 2, startcol=4, header=False)
    writer.close()
    processed_data = output.getvalue()
    return processed_data


def get_grouped_figure_test(dfm1, dfm2, G1, G2, g01, g02, day_range=[6, 18]):
    data1 = dfm1.copy()
    data2 = dfm2.copy()
    df = data1 if len(data1) > len(data2) else data2
    trace_data = []

    if (len(G1) == 0) and (len(G2) == 0):
        trace_data.append(go.Scatter(x=list(df.index.values + 1), y=[]))
    elif (len(G2) != 0) or (len(G1) != 0):
        df1 = get_grouped_data(data1, G1)
        df2 = get_grouped_data(data2, G2)
        hover_text = None
        if sum(df1.Mean != 0) != 0:
            if isinstance(df1.DateTime.iloc[0], pd.Period):
                df1.DateTime = df1.DateTime.apply(pd.Period.to_timestamp)

            trace_data.append(go.Scatter(x=df1.DateTime, y=df1.Upper_Dev,
                                         fill=None,
                                         mode='lines', fillcolor="rgba(27, 67, 50, 0.4)",
                                         line=dict(color='rgba(255,255,255,0)'),
                                         showlegend=False, name="+ dev"
                                         ))
            trace_data.append(go.Scatter(
                x=df1.DateTime,
                y=df1.Lower_Dev,
                name="- dev",
                fill='tonexty',  # fill area between trace0 and trace1
                mode='lines', fillcolor="rgba(27, 67, 50, 0.4)", line=dict(color='rgba(255,255,255,0)'),
                showlegend=False))
            trace_data.append(
                go.Scatter(x=df1.DateTime, y=df1.Mean,
                           fill=None,
                           mode='lines', line=dict(color="rgba(27, 67, 50, 1)"),
                           name=g01)
            )

        if sum(df2.Mean != 0) != 0:
            if isinstance(df2.DateTime.iloc[0], pd.Period):
                df2.DateTime = df2.DateTime.apply(pd.Period.to_timestamp)
            trace_data.append(go.Scatter(x=df2.DateTime, y=df2.Upper_Dev,
                                         fill=None,
                                         mode='lines',
                                         line=dict(
                                             color='rgba(255,255,255,0)'),
                                         showlegend=False, name="+ dev"
                                         ))
            trace_data.append(go.Scatter(
                x=df2.DateTime,
                y=df2.Lower_Dev,
                fill='tonexty',  # fill area between trace0 and trace1
                mode='lines',
                fillcolor="rgba(164, 19, 60, 0.1)", line=dict(color='rgba(255,255,255,0)'),
                name="- dev",
                showlegend=False))
            trace_data.append(
                go.Scatter(x=df2.DateTime, y=df2.Mean,
                           fill=None,
                           mode='lines', line=dict(color="rgba(164, 19, 60, 1)"),
                           name=g02)
            )

    fig = go.Figure(data=trace_data)
    # if isinstance(data.DateTime.iloc[0], pd.Period):
    #     data.DateTime = data.DateTime.apply(pd.Period.to_timestamp)
    if ~isinstance(data1.DateTime.iloc, pd.Timestamp):
        data1.DateTime = data1.DateTime.apply(pd.to_datetime)
    data1["hour"] = data1.DateTime.dt.hour
    if (data1.DateTime.dt.hour == 0).sum() != data1.shape[0]:
        data1["timeDay"] = data1["hour"].apply(lambda x: "day" if x in list(
            range(day_range[0] + 1, day_range[1])) else "night")
        ind = data1.DateTime
        vals = data1.timeDay.values
        prev = 0
        times = []
        for i in range(1, data1.timeDay.shape[0] - 1):
            if vals[i] != vals[prev]:
                if vals[prev] == "night":
                    times.append((ind.iloc[prev], ind.iloc[i - 1]))
                prev = i
        if (vals[-1] == "night"):
            times.append((ind.iloc[prev], ind.iloc[-1]))

        for time in times:
            fig.add_vrect(
                x0=time[0],
                x1=time[1],
                fillcolor="black",
                opacity=0.3,
                line_width=0,
            )
    if ~isinstance(data2.DateTime.iloc, pd.Timestamp):
        data2.DateTime = data2.DateTime.apply(pd.to_datetime)
    data2["hour"] = data2.DateTime.dt.hour
    if (data2.DateTime.dt.hour == 0).sum() != data2.shape[0]:
        data2["timeDay"] = data2["hour"].apply(lambda x: "day" if x in list(
            range(day_range[0] + 1, day_range[1])) else "night")
        ind = data2.DateTime
        vals = data2.timeDay.values
        prev = 0
        times = []
        for i in range(1, data2.timeDay.shape[0] - 1):
            if vals[i] != vals[prev]:
                if vals[prev] == "night":
                    times.append((ind.iloc[prev], ind.iloc[i - 1]))
                prev = i
        if (vals[-1] == "night"):
            times.append((ind.iloc[prev], ind.iloc[-1]))

        for time in times:
            fig.add_vrect(
                x0=time[0],
                x1=time[1],
                fillcolor="black",
                opacity=0.3,
                line_width=0,
            )
    fig.update_layout(hovermode="x unified", yaxis_title="Value", xaxis_title="Time",
                      hoverlabel=dict(bgcolor="gray", font_color="black", font_size=16,
                                      font_family="Rockwell"),
                      xaxis=dict(rangeslider=dict(visible=True))
                      )

    return fig
