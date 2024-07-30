Features

Sales Trend Analysis: Visualize sales trends over specified time periods and determine if the trend is positive or negative.
Store Performance Analysis: Compare sales data across all stores and highlight the top-performing stores.
Negative Trend Stores: Identify stores with negative sales trends and calculate their percentage loss.
Promotions Impact Analysis: Visualize the impact of various promotions on transaction counts and net sales, categorized by promotion items and platforms.
Derivative Analysis: Analyze the first derivative of transaction counts and net sales to understand the rate of change during promotional periods.
Setup and Installation

Prerequisites
Python 3.7 or higher
pip (Python package installer)

Installation
Clone the repository:
https://github.com/your-username/trilytics.git
Navigate to the project directory:

cd trilytics

Install the required dependencies:

pip install -r requirements.txt

Running the Application
To run the Streamlit application, use the following command:

streamlit run app.py

Usage

Once the application is running, navigate to the provided local URL in your web browser. The application consists of multiple sections:

Exploratory Data Analysis:

View sales trends for specified periods.
Analyze store-wise performance and identify top-performing stores.
Examine stores with negative sales trends and their percentage loss.
Promotions Analysis:

Visualize the impact of different promotions on transaction counts and net sales.
Explore the rate of change during promotional periods through derivative analysis.
File Structure
app.py: Main script for the Streamlit application.
data/: Directory containing CSV data files used for analysis.
requirements.txt: List of Python packages required for the project.
Data Sources
The data used in this project includes:

sales_month_wise_90.csv: Contains sales data for a 3-month period.
sales_month_wise_7.csv: Contains sales data for a 1-week period.
sales_store_wise_90.csv: Store-wise sales data over a 3-month period.
promotions_final.csv: Information about promotions including start and end dates, promotion items, and platforms.