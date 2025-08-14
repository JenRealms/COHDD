from dotenv import load_dotenv
import pandas as pd
import io
import re
import logging
import boto3
import os
import time
import sys
import os

from .input_sql_query import get_station_sql_query

load_dotenv()

class GetStationData:
    def __init__(self):
        self.s3_output_location = 's3://orbit-science-ai-assist/COH-Data/athena-output/'
        self.athena_client = boto3.client('athena')

    def create_markdown_table(self, headers, rows):
        """
        Create a markdown table from headers and rows
        """

        if not rows:
            return "No data found"
    
        # Calculate maximum width for each column
        widths = [len(str(header)) for header in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))
    
        # Create header row
        markdown = '| ' + ' | '.join(str(header).ljust(width) for header, width in zip(headers, widths)) + ' |\n'
        # Create separator row
        markdown += '|' + '|'.join('-' * (width + 2) for width in widths) + '|\n'
        # Create data rows
        for row in rows[1:]:
            markdown += '| ' + ' | '.join(str(cell).ljust(width) for cell, width in zip(row, widths)) + ' |\n'
    
        return markdown
    
    def convert_markdown_2_df(self, markdown_table: str) -> pd.DataFrame:
        """
        Convert the table in the markdown format to a pandas DataFrame
        """
        try:
                # Remove separator lines (e.g., | --- | --- |) using regex
            lines = markdown_table.strip().split('\n')
            clean_lines = [line for line in lines if not re.match(r'^\s*\|?[\s\-|:]+\|?\s*$', line)]

            # Combine into a CSV-like string
            csv_text = '\n'.join(clean_lines)

            #print(csv_text)
            
            # Convert markdown table to DataFrame using `read_csv`
            df = pd.read_csv(io.StringIO(csv_text), sep='|').dropna(axis=1, how='all')
            #print(df.head(10))
            # Strip whitespace from column headers
            df.columns = df.columns.str.strip()
            
            # Strip whitespace from string entries
            df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
            
            # Convert all columns to numeric EXCEPT station_code and ofd_date
            for col in df.columns:
                if col not in ['station_code', 'ofd_date']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
        except Exception as e:
            logging.error(f"Error converting markdown table to pandas DataFrame: {str(e)}")
            print(f"Error converting markdown table to pandas DataFrame: {str(e)}")
            return False 
        
    def query_athena_to_dataframe(self, sql_query):
        """
        Execute SQL query on Athena and return results as markdown formatted string

        Parameters:
        ------------
        sql_query: str
            SQL query to execute
        s3_output_location: str
        S3 bucket location where Athena will store query results
        athena_client: boto3.client('athena')
            Athena client
        
        Returns:
        --------
        str
        Query results formatted as markdown table
        """

        # Start query execution
        response = self.athena_client.start_query_execution(
            QueryString=sql_query,
            QueryExecutionContext={
                'Database': 'coh_v3'
            },
            ResultConfiguration={
                'OutputLocation': self.s3_output_location,
            }
        )
        
        #print(response)

        # Get query execution ID
        query_execution_id = response['QueryExecutionId']
        
        # Wait for query to complete
        status = 'RUNNING'
        while status in ['RUNNING', 'QUEUED']:
            response = self.athena_client.get_query_execution(
                QueryExecutionId=query_execution_id
            )
            status = response['QueryExecution']['Status']['State']
            
            if status in ['RUNNING', 'QUEUED']:
                time.sleep(1)
        
        # Check if query executed successfully
        if status == 'SUCCEEDED':
            # Get results using paginator
            results_paginator = self.athena_client.get_paginator('get_query_results')
            results_iter = results_paginator.paginate(
                QueryExecutionId=query_execution_id
            )
            
            # Process results
            data_rows = []
            column_names = []
            
            # Process result pages
            for page in results_iter:
                # Extract column names from first page
                if not column_names:
                    column_info = page['ResultSet']['ResultSetMetadata']['ColumnInfo']
                    column_names = [col['Label'] for col in column_info]
                
                # Process rows (skip header row in first page)
                rows = page['ResultSet']['Rows']
                if page is results_iter and len(rows) > 0:
                    rows = rows[1:]  # Skip header row in first page
                    
                # Extract the data
                for row in rows:
                    data_row = [field.get('VarCharValue', '') if field else '' 
                               for field in row['Data']]
                    data_rows.append(data_row)
            
            # Convert to markdown format
            #print(self.create_markdown_table(column_names, data_rows))
            df = self.convert_markdown_2_df(self.create_markdown_table(column_names, data_rows))
            return df
            
        else:
            error_msg = response['QueryExecution']['Status'].get(
                'StateChangeReason', 'Unknown error')
            raise Exception(f"Query failed: {error_msg}")
        
    def get_sql_query(self, station_code, target_date):
        sql_query = get_station_sql_query(station_code, target_date)
        return sql_query
    
    def get_station_data(self, station_code, target_date):
        sql_query = self.get_sql_query(station_code, target_date)
        return self.query_athena_to_dataframe(sql_query)
    

get_data = GetStationData()

#print(get_date.get_sql_query('DYR3', '2025-07-21'))
#print("*"*80)
#print(get_date.get_station_data('DYR3', '2025-07-21').iloc[0, 2])
#print("*"*80)
#print("*"*80)
#print(get_data.get_station_data('DYR3', '2025-08-12').iloc[0, 3:])
#print("*"*80)