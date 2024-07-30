import streamlit as st
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from datetime import datetime
import subprocess

# def run_script():
#     subprocess.run(["python", "main.py"], capture_output=True, text=True)
#     # st.write(result.stdout)



st.set_page_config(layout="wide")
st.title("Trilytics") 

# if st.button('Run External Script'):
#     run_script()

st.subheader("Exploratory Data Analysis")




#################################################################################
# Sales

def plot_sales_trend(file_path, title):
    df = pd.read_csv(file_path, parse_dates=['START_DATE'], dayfirst=True)
    df['START_DATE'] = pd.to_datetime(df['START_DATE'], format='%d-%m-%Y')
    
    X = np.array(range(len(df['START_DATE']))).reshape(-1, 1)
    y = df['NET_SALES_FINAL_USD_AMOUNT'].values
    
    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    trend = "Positive" if slope > 0 else "Negative"
    
    fig = make_subplots(rows=1, cols=1)
    
    fig.add_trace(
        go.Scatter(
            x=df['START_DATE'],
            y=df['NET_SALES_FINAL_USD_AMOUNT'],
            mode='lines+markers',
            name='Net Sales'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['START_DATE'],
            y=model.predict(X),
            mode='lines',
            name='Trend Line',
            line=dict(dash='dash', color='blue')
        )
    )
    
    trend_text = f'Sales Trend: {trend}'
    fig.add_annotation(
        x=df['START_DATE'].iloc[int(len(df) * 0.7)],
        y=max(df['NET_SALES_FINAL_USD_AMOUNT']) * 0.95,
        text=trend_text,
        showarrow=False,
        font=dict(size=12, color='red')
    )
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Net Sales (USD)',
        xaxis_tickangle=-45,
        legend=dict(x=0, y=1),
        height=600,
        width=1200,
        template='plotly_white'
    )
    
    return fig

# Usage in Streamlit
st.plotly_chart(plot_sales_trend('sales_month_wise_90.csv', '3 month sales'))
st.plotly_chart(plot_sales_trend('sales_month_wise_7.csv', '1 week sales'))




################################################################################
#Store wise
# Read data from CSV file
df = pd.read_csv('sales_store_wise_90.csv', parse_dates=['START_DATE'], dayfirst=True)

# Convert START_DATE to datetime
df['START_DATE'] = pd.to_datetime(df['START_DATE'], format='%d-%m-%Y')

# Get unique store numbers
unique_stores = df['STORE_NUMBER'].unique()

# List to hold store numbers and their corresponding slopes
store_slopes = []

# Create Plotly figure for all stores
fig_all_stores = go.Figure()

for store in unique_stores:
    df_store = df[df['STORE_NUMBER'] == store]

    # Prepare data for linear regression
    X = np.array(range(len(df_store))).reshape(-1, 1)
    y = df_store['NET_SALES_FINAL_USD_AMOUNT'].values

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Get the slope (coefficient)
    slope = model.coef_[0]
    store_slopes.append((store, slope))

    # Plot the net sales for the store
    fig_all_stores.add_trace(go.Scatter(
        x=df_store['START_DATE'],
        y=df_store['NET_SALES_FINAL_USD_AMOUNT'],
        mode='lines+markers',
        name=f'Store {store}'
    ))

fig_all_stores.update_layout(
    title='Net Sales over Time for All Stores',
    xaxis_title='Date',
    yaxis_title='Net Sales (USD)',
    xaxis_tickangle=-45,
    template='plotly_white',
    width=1200,
    height=600
)

# Sort the stores by slope in descending order
store_slopes.sort(key=lambda x: x[1], reverse=True)

# Get the top 10 stores with the highest slopes
top_10_stores = store_slopes[:10]

# Create Plotly figure for top 10 stores
fig_top_10_stores = go.Figure()

for store, slope in top_10_stores:
    df_store = df[df['STORE_NUMBER'] == store]
    fig_top_10_stores.add_trace(go.Scatter(
        x=df_store['START_DATE'],
        y=df_store['NET_SALES_FINAL_USD_AMOUNT'],
        mode='lines+markers',
        name=f'Store {store}'
    ))

fig_top_10_stores.update_layout(
    title='Net Sales over Time for Top 10 Stores with Maximum Returns',
    xaxis_title='Date',
    yaxis_title='Net Sales (USD)',
    xaxis_tickangle=-45,
    template='plotly_white',
    width=1200,
    height=600
)

# Display the figures in Streamlit
st.plotly_chart(fig_all_stores)
st.plotly_chart(fig_top_10_stores)

# Print the top 10 stores with maximum returns
# st.write("Top 10 stores with maximum returns (highest slopes):")
# for store, slope in top_10_stores:
#     st.write(f"Store {store}: Slope {slope}")


# Read data from CSV file
df = pd.read_csv('sales_store_wise_90.csv', parse_dates=['START_DATE'], dayfirst=True)

# Convert START_DATE to datetime
df['START_DATE'] = pd.to_datetime(df['START_DATE'], format='%d-%m-%Y')

# Get unique store numbers
unique_stores = df['STORE_NUMBER'].unique()

# List to hold stores with negative trends and their percentage loss
negative_trend_stores = []

# Calculate percentage loss for stores with negative trends
for store in unique_stores:
    df_store = df[df['STORE_NUMBER'] == store]

    # Prepare data for linear regression
    X = np.array(range(len(df_store))).reshape(-1, 1)  # Convert index to array for regression
    y = df_store['NET_SALES_FINAL_USD_AMOUNT'].values

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Get the slope (coefficient)
    slope = model.coef_[0]

    # Check if slope is negative
    if slope < 0:
        # Calculate percentage loss from the first to the last sale
        initial_sales = df_store['NET_SALES_FINAL_USD_AMOUNT'].iloc[0]
        final_sales = df_store['NET_SALES_FINAL_USD_AMOUNT'].iloc[-1]

        if initial_sales > 0:
            percentage_loss = ((initial_sales - final_sales) / initial_sales) * 100

            # Only append if the percentage loss is positive
            if percentage_loss > 0:
                negative_trend_stores.append((store, percentage_loss))

# Create a DataFrame for negative trends
negative_trends_df = pd.DataFrame(negative_trend_stores, columns=['Store Number', 'Percentage Loss'])

# Plotting using Plotly
fig = go.Figure()

fig.add_trace(go.Bar(
    x=negative_trends_df['Store Number'],
    y=negative_trends_df['Percentage Loss'],
    text=negative_trends_df['Percentage Loss'].apply(lambda x: f'{x:.2f}%'),
    textposition='outside',
    marker=dict(color='red')
))

fig.update_layout(
    title='Stores with Negative Trends and Their Percentage Loss',
    xaxis_title='Store Number',
    yaxis_title='Percentage Loss',
    xaxis=dict(tickangle=-45),
    template='plotly_white',
    width=1200,
    height=600
)

# Display the DataFrame and the figure in Streamlit
st.write("**Stores with Negative Trends and Their Percentage Loss:**")
st.write(negative_trends_df)
# st.plotly_chart(fig)




##################################################################################
#Promotions

# Load data
promotions_data_frame = pd.read_csv('promotions_final.csv')
df_monthwise_rnn = pd.read_csv('sales_month_wise_1.csv')

# Convert date columns to datetime objects
promotions_data_frame['PROMOTION_START_DATE'] = pd.to_datetime(promotions_data_frame['PROMOTION_START_DATE'], format='%d/%m/%Y')
promotions_data_frame['PROMOTION_END_DATE'] = pd.to_datetime(promotions_data_frame['PROMOTION_END_DATE'], format='%d/%m/%Y')

# Create a date range for the x-axis
start_date = datetime(2021, 4, 1)
end_date = datetime(2023, 12, 31)

# Create Plotly figures
fig1 = go.Figure()
fig2 = go.Figure()
fig3 = go.Figure()
fig4 = go.Figure()

# Define colors for promotions and platforms
promotion_colors = {
    'Delivery Fee': 'lightblue',
    'Chicken Sandwich': 'lightgreen',
    'Fries': 'lightcoral',
    'Sandwich': 'yellow',
    'Brownie': 'lightpink',
    'Chicken Sandwich Combo': 'purple',
    'Corn': 'brown',
    'Overall Discount': 'lightgray'
}

platform_colors = {
    'DD': 'lightblue',
    'Inhouse': 'lightgreen',
    'UE': 'lightcoral'
}

# Plot promotions and data for "TRANSACTION_FINAL_COUNT" and "NET_SALES_FINAL_USD_AMOUNT"
for _, row in promotions_data_frame.iterrows():
    start = row['PROMOTION_START_DATE']
    end = row['PROMOTION_END_DATE']
    promotion_type = row['PROMOTION_ITEM']
    
    color = promotion_colors.get(promotion_type, 'lightgray')
    fig1.add_shape(type="rect",
                   x0=start, x1=end,
                   y0=0, y1=140000,
                   fillcolor=color, 
                   opacity=0.5,
                   line=dict(color="rgba(255,255,255,0)"))
    
    fig2.add_shape(type="rect",
                   x0=start, x1=end,
                   y0=0, y1=5e6,
                   fillcolor=color, 
                   opacity=0.5,
                   line=dict(color="rgba(255,255,255,0)"))

for _, row in promotions_data_frame.iterrows():
    start = row['PROMOTION_START_DATE']
    end = row['PROMOTION_END_DATE']
    platform_type = row['PLATFORM']
    
    color = platform_colors.get(platform_type, 'lightgray')
    fig3.add_shape(type="rect",
                   x0=start, x1=end,
                   y0=0, y1=140000,
                   fillcolor=color, 
                   opacity=0.5,
                   line=dict(color="rgba(255,255,255,0)"))
    
    fig4.add_shape(type="rect",
                   x0=start, x1=end,
                   y0=0, y1=5e6,
                   fillcolor=color, 
                   opacity=0.5,
                   line=dict(color="rgba(255,255,255,0)"))

# Add lines for "TRANSACTION_FINAL_COUNT" and "NET_SALES_FINAL_USD_AMOUNT"
fig1.add_trace(go.Scatter(
    x=pd.to_datetime(df_monthwise_rnn['START_DATE'], format='%d-%m-%Y'),
    y=df_monthwise_rnn["TRANSACTION_FINAL_COUNT"],
    mode='lines',
    name='Original Data',
    line=dict(color='black')
))

fig2.add_trace(go.Scatter(
    x=pd.to_datetime(df_monthwise_rnn['START_DATE'], format='%d-%m-%Y'),
    y=df_monthwise_rnn["NET_SALES_FINAL_USD_AMOUNT"],
    mode='lines',
    name='Original Data',
    line=dict(color='black')
))

fig3.add_trace(go.Scatter(
    x=pd.to_datetime(df_monthwise_rnn['START_DATE'], format='%d-%m-%Y'),
    y=df_monthwise_rnn["TRANSACTION_FINAL_COUNT"],
    mode='lines',
    name='Original Data',
    line=dict(color='black')
))

fig4.add_trace(go.Scatter(
    x=pd.to_datetime(df_monthwise_rnn['START_DATE'], format='%d-%m-%Y'),
    y=df_monthwise_rnn["NET_SALES_FINAL_USD_AMOUNT"],
    mode='lines',
    name='Original Data',
    line=dict(color='black')
))

for promotion, color in promotion_colors.items():
    fig1.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(size=10, color=color),
        legendgroup=promotion,
        showlegend=True,
        name=promotion
    ))

for promotion, color in promotion_colors.items():
    fig2.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(size=10, color=color),
        legendgroup=promotion,
        showlegend=True,
        name=promotion
    ))

for platform, color in platform_colors.items():
    fig3.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(size=10, color=color),
        legendgroup=platform,
        showlegend=True,
        name=platform
    ))

for platform, color in platform_colors.items():
    fig4.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(size=10, color=color),
        legendgroup=platform,
        showlegend=True,
        name=platform
    ))

# Update layout for all figures
for fig in [fig1, fig2, fig3, fig4]:
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Value',
        xaxis=dict(
            range=[start_date, end_date],
            tickformat='%b %Y'
        ),
        template='plotly_white'
    )

fig1.update_layout(
    title='Promotion Item Classification of "TRANSACTION_FINAL_COUNT"',
    annotations=[go.layout.Annotation(
        # text='Promotion Item Color Legend: Delivery Fee (lightblue), Chicken Sandwich (lightgreen), Fries (lightcoral), Sandwich (yellow), Brownie (lightpink), Chicken Sandwich Combo (purple), Corn (brown), Overall Discount (lightgray)',
        xref='paper', yref='paper',
        x=0, y=-0.2,
        showarrow=False
    )]
)

fig2.update_layout(
    title='Promotion Item Classification of "NET_SALES_FINAL_USD_AMOUNT"',
    annotations=[go.layout.Annotation(
        # text='Promotion Item Color Legend: Delivery Fee (lightblue), Chicken Sandwich (lightgreen), Fries (lightcoral), Sandwich (yellow), Brownie (lightpink), Chicken Sandwich Combo (purple), Corn (brown), Overall Discount (lightgray)',
        xref='paper', yref='paper',
        x=0, y=-0.2,
        showarrow=False
    )]
)

fig3.update_layout(
    title='Promotion Platform Classification of "TRANSACTION_FINAL_COUNT"',
    annotations=[go.layout.Annotation(
        # text='Promotion Platform Color Legend: DD (lightblue), Inhouse (lightgreen), UE (lightcoral)',
        xref='paper', yref='paper',
        x=0, y=-0.2,
        showarrow=False
    )]
)

fig4.update_layout(
    title='Promotion Platform Classification of "NET_SALES_FINAL_USD_AMOUNT"',
    annotations=[go.layout.Annotation(
        # text='Promotion Platform Color Legend: DD (lightblue), Inhouse (lightgreen), UE (lightcoral)',
        xref='paper', yref='paper',
        x=0, y=-0.2,
        showarrow=False
    )]
)

# Display the figures in Streamlit
st.plotly_chart(fig1)
st.plotly_chart(fig2)
st.plotly_chart(fig3)
st.plotly_chart(fig4)




def analyse_promootion_derivative():
    promotions_data_frame = pd.read_csv('promotions_final.csv')
    df_monthwise_rnn = pd.read_csv('sales_month_wise_1.csv')

    # Convert date columns to datetime objects
    promotions_data_frame['PROMOTION_START_DATE'] = pd.to_datetime(promotions_data_frame['PROMOTION_START_DATE'], format='%d/%m/%Y')
    promotions_data_frame['PROMOTION_END_DATE'] = pd.to_datetime(promotions_data_frame['PROMOTION_END_DATE'], format='%d/%m/%Y')
    df_monthwise_rnn['START_DATE'] = pd.to_datetime(df_monthwise_rnn['START_DATE'], format='%d-%m-%Y')

    # Calculate the derivatives
    df_monthwise_rnn['DERIVATIVE_TRANSACTION'] = df_monthwise_rnn['TRANSACTION_FINAL_COUNT'].diff()
    df_monthwise_rnn['DERIVATIVE_AMOUNT'] = df_monthwise_rnn['NET_SALES_FINAL_USD_AMOUNT'].diff()

    # Create a date range for the x-axis
    start_date = datetime(2021, 4, 1)
    end_date = datetime(2023, 12, 31)

    # Create Plotly figures
    fig1 = go.Figure()
    fig2 = go.Figure()
    fig3 = go.Figure()
    fig4 = go.Figure()

    # Define colors for promotions and platforms
    promotion_colors = {
        'Delivery Fee': 'lightblue',
        'Chicken Sandwich': 'lightgreen',
        'Fries': 'lightcoral',
        'Sandwich': 'yellow',
        'Brownie': 'lightpink',
        'Chicken Sandwich Combo': 'purple',
        'Corn': 'brown',
        'Overall Discount': 'lightgray'
    }

    platform_colors = {
        'DD': 'lightblue',
        'Inhouse': 'lightgreen',
        'UE': 'lightcoral'
    }

    # Plot promotions and data for "DERIVATIVE_TRANS" and "DERIVATIVE_AMT"
    for _, row in promotions_data_frame.iterrows():
        start = row['PROMOTION_START_DATE']
        end = row['PROMOTION_END_DATE']
        promotion_type = row['PROMOTION_ITEM']
        
        color = promotion_colors.get(promotion_type, 'lightgray')
        fig1.add_shape(type="rect",
                       x0=start, x1=end,
                       y0=df_monthwise_rnn['DERIVATIVE_TRANSACTION'].min(), y1=df_monthwise_rnn['DERIVATIVE_TRANSACTION'].max(),
                       fillcolor=color, 
                       opacity=0.5,
                       line=dict(color="rgba(255,255,255,0)"))
        
        fig2.add_shape(type="rect",
                       x0=start, x1=end,
                       y0=df_monthwise_rnn['DERIVATIVE_AMOUNT'].min(), y1=df_monthwise_rnn['DERIVATIVE_AMOUNT'].max(),
                       fillcolor=color, 
                       opacity=0.5,
                       line=dict(color="rgba(255,255,255,0)"))

    for _, row in promotions_data_frame.iterrows():
        start = row['PROMOTION_START_DATE']
        end = row['PROMOTION_END_DATE']
        platform_type = row['PLATFORM']
        
        color = platform_colors.get(platform_type, 'lightgray')
        fig3.add_shape(type="rect",
                       x0=start, x1=end,
                       y0=df_monthwise_rnn['DERIVATIVE_TRANSACTION'].min(), y1=df_monthwise_rnn['DERIVATIVE_TRANSACTION'].max(),
                       fillcolor=color, 
                       opacity=0.5,
                       line=dict(color="rgba(255,255,255,0)"))
        
        fig4.add_shape(type="rect",
                       x0=start, x1=end,
                       y0=df_monthwise_rnn['DERIVATIVE_AMOUNT'].min(), y1=df_monthwise_rnn['DERIVATIVE_AMOUNT'].max(),
                       fillcolor=color, 
                       opacity=0.5,
                       line=dict(color="rgba(255,255,255,0)"))

    # Add lines for "DERIVATIVE_TRANS" and "DERIVATIVE_AMT"
    fig1.add_trace(go.Scatter(
        x=df_monthwise_rnn['START_DATE'],
        y=df_monthwise_rnn["DERIVATIVE_TRANSACTION"],
        mode='lines',
        name='Derivative of Transaction Count',
        line=dict(color='black')
    ))

    fig2.add_trace(go.Scatter(
        x=df_monthwise_rnn['START_DATE'],
        y=df_monthwise_rnn["DERIVATIVE_AMOUNT"],
        mode='lines',
        name='Derivative of Net Sales Amount',
        line=dict(color='black')
    ))

    fig3.add_trace(go.Scatter(
        x=df_monthwise_rnn['START_DATE'],
        y=df_monthwise_rnn["DERIVATIVE_TRANSACTION"],
        mode='lines',
        name='Derivative of Transaction Count',
        line=dict(color='black')
    ))

    fig4.add_trace(go.Scatter(
        x=df_monthwise_rnn['START_DATE'],
        y=df_monthwise_rnn["DERIVATIVE_AMOUNT"],
        mode='lines',
        name='Derivative of Net Sales Amount',
        line=dict(color='black')
    ))

    for promotion, color in promotion_colors.items():
        fig1.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            legendgroup=promotion,
            showlegend=True,
            name=promotion
        ))

    for promotion, color in promotion_colors.items():
        fig2.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            legendgroup=promotion,
            showlegend=True,
            name=promotion
        ))

    for platform, color in platform_colors.items():
        fig3.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            legendgroup=platform,
            showlegend=True,
            name=platform
        ))

    for platform, color in platform_colors.items():
        fig4.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            legendgroup=platform,
            showlegend=True,
            name=platform
        ))

    # Update layout for all figures
    for fig in [fig1, fig2, fig3, fig4]:
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Value',
            xaxis=dict(
                range=[start_date, end_date],
                tickformat='%b %Y'
            ),
            template='plotly_white'
        )

    fig1.update_layout(
        title='Promotion Item Classification of "DERIVATIVE_TRANSACTION"',
        annotations=[go.layout.Annotation(
            xref='paper', yref='paper',
            x=0, y=-0.2,
            showarrow=False
        )]
    )

    fig2.update_layout(
        title='Promotion Item Classification of "DERIVATIVE_AMOUNT"',
        annotations=[go.layout.Annotation(
            xref='paper', yref='paper',
            x=0, y=-0.2,
            showarrow=False
        )]
    )

    fig3.update_layout(
        title='Promotion Platform Classification of "DERIVATIVE_TRANSACTION"',
        annotations=[go.layout.Annotation(
            xref='paper', yref='paper',
            x=0, y=-0.2,
            showarrow=False
        )]
    )

    fig4.update_layout(
        title='Promotion Platform Classification of "DERIVATIVE_AMOUNT"',
        annotations=[go.layout.Annotation(
            xref='paper', yref='paper',
            x=0, y=-0.2,
            showarrow=False
        )]
    )

    # Display the figures in Streamlit
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)
    st.plotly_chart(fig3)
    st.plotly_chart(fig4)


analyse_promootion_derivative()
