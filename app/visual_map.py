import os
import streamlit as st

from google.cloud import bigquery

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/opt/ml/essential_config/aitech4-finalproject-0000-b7e0b3c932a1.json'
bigquery_client = bigquery.Client()

sql = f"""
SELECT
    bbox, lat, lon
FROM
    `aitech4-finalproject-0000.pothole_serving_log.pothole_detection`
"""

query_job = bigquery_client.query(sql)
df = query_job.to_dataframe()
print(df)

# st.set_page_config(layout="wide")
# st.title("Potholes on Map")


# def main():
#     pass
    