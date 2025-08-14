import boto3
import time
import pprint
from pydantic import BaseModel, Field
from langchain_core.tools import tool, StructuredTool
from langgraph.prebuilt import ToolNode

@tool
def query_athena_to_markdown(sql_query: str):
        """
        Execute SQL query on Athena and return results as markdown formatted string
        
        Parameters:
        -----------
        sql_query: str
            SQL query to execute
        s3_output_location: str
            S3 bucket location where Athena will store query results
            
        Returns:
        --------
        str
            Query results formatted as markdown table
        """
        # Initialize Athena client
        athena_client = boto3.client('athena')
        # the location where the query results will be stored
        s3_output_location='s3://orbit-science-ai-assist/COH-Data/athena-output/'
        
        try:
            # Start query execution
            response = athena_client.start_query_execution(
                QueryString=sql_query,
                QueryExecutionContext={
                    'Database': 'coh_v3'
                },
                ResultConfiguration={
                    'OutputLocation': s3_output_location,
                }
            )
            
            # Get query execution ID
            query_execution_id = response['QueryExecutionId']
            
            # Wait for query to complete
            status = 'RUNNING'
            while status in ['RUNNING', 'QUEUED']:
                response = athena_client.get_query_execution(
                    QueryExecutionId=query_execution_id
                )
                status = response['QueryExecution']['Status']['State']
                
                if status in ['RUNNING', 'QUEUED']:
                    time.sleep(1)
            
            # Check if query executed successfully
            if status == 'SUCCEEDED':
                # Get results using paginator
                results_paginator = athena_client.get_paginator('get_query_results')
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
                return create_markdown_table(column_names, data_rows)
                
            else:
                error_msg = response['QueryExecution']['Status'].get(
                    'StateChangeReason', 'Unknown error')
                raise Exception(f"Query failed: {error_msg}")
                
        except Exception as e:
            print(f"Error executing Athena query: {str(e)}")
            raise
        
def create_markdown_table(headers, rows):
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
    
#tool = ToolNode(
#     [
#          StructuredTool.from_function(
#               func=query_athena_to_markdown,
#               name=query_athena_to_markdown.__name__,
#               #description="Execute SQL query on Athena and return results as markdown formatted string"
#          )
#     ]
#)


#if __name__ == "__main__":
#    tool = query_athena_to_markdown()
#    sql_query = """
#        SELECT * FROM "AWSDataCatalog"."ai-assist"."coh_v3_prod" LIMIT 10
#     """
#    print(tool.query_athena_to_markdown(sql_query))