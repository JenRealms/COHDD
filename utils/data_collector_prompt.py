"""
data_collector_prompt.py

This module defines the DataCollectorPrompt class, which constructs prompts for data collection tasks
using AWS Athena and S3 resources. Handles loading of prompt resources and prompt template construction.
"""
from .S3_reader import S3_reader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import dotenv
dotenv.load_dotenv()

class DataCollectorPrompt():
    """
    Constructs prompts for data collection tasks, loading necessary resources from S3.
    """
    def __init__(self):
        """
        Initializes the DataCollectorPrompt by loading required resources from S3.
        Handles errors during S3 reads and logs them.
        """
        try:
            self.rc_plan = S3_reader()._run('s3://orbit-science-ai-assist/COH-Data/coh_rc_plan.txt')
        except Exception as e:
            print(f"[ERROR] Failed to load rc_plan from S3: {e}")
            self.rc_plan = None
        try:
            self.rc_understanding = S3_reader()._run('s3://orbit-science-ai-assist/COH-Data/Understanding_COH_Root_Cause.txt')
        except Exception as e:
            print(f"[ERROR] Failed to load rc_understanding from S3: {e}")
            self.rc_understanding = None
        try:
            self.column_definitions = S3_reader()._run('s3://orbit-science-ai-assist/COH-Data/coh_v3_column_definition.txt')
        except Exception as e:
            print(f"[ERROR] Failed to load column_definitions from S3: {e}")
            self.column_definitions = None
        try:
            self.schema = S3_reader()._run('s3://orbit-science-ai-assist/COH-Data/schema.txt')
        except Exception as e:
            print(f"[ERROR] Failed to load schema from S3: {e}")
            self.schema = None
        try:
            self.sql_examples = S3_reader()._run('s3://orbit-science-ai-assist/COH-Data/examples.txt')
        except Exception as e:
            print(f"[ERROR] Failed to load sql_examples from S3: {e}")
            self.sql_examples = None
        try:
            self.guidelines = S3_reader()._run('s3://orbit-science-ai-assist/COH-Data/guidance.txt')
        except Exception as e:
            print(f"[ERROR] Failed to load guidelines from S3: {e}")
            self.guidelines = None
        try:
            self.coh_business = S3_reader()._run('s3://orbit-science-ai-assist/COH-Data/guidance.txt')
        except Exception as e:
            print(f"[ERROR] Failed to load coh_business from S3: {e}")
            self.coh_business = None

    def get_system_prompt(self):
        """
        Returns the system prompt for the data analyst agent.
        """
        system_prompt = """
            You are a senior data analyst known for excellent SQL skills and a deep understanding of the AWS Athena database. 
            You are also known for your ability to write clear and concise SQL queries.
            You will write **ANSI SQL** queries to get the data from the AWS Athena database.
        """
        return system_prompt

    def get_task_prompt(self, schema, column_definitions, sql_examples, guidelines):
        """
        Constructs the task prompt using schema, column definitions, SQL examples, and guidelines.
        """
        task_prompt =  """
        You will be given a data collection request as follows and you will generate a SQL query which will be fed into the next tool to get the data.
        <Data Collection Request>
            {task}
        </Data Collection Request>
        After receiving the data request from the manager or other agents, you will:
                1. Generate an appropriate SQL query
                    - Use the database schema and column definitions to generate the query
                      -- Here is the database schema:
                        <database schema>
                            {schema}
                        </database schema>
                    -- Here are the column definitions:
                        <column definitions>
                            {column_definitions}
                        </column definitions>
                    -- Here are the examples for your references:
                        <sql examples>
                            {sql_examples}
                        </sql examples>
                    -- Note that the database in the AWS Athena is called "AwsDataCatalog"."ai-assist"."coh__v3_prod"
                    -- Note that the schema and columns listed above are what are currently available in the database.
                2. Follow the guidelines to generate the sql query
                    <guidelines>
                        {guidelines}
                    </guidelines>
                3. All queries will follow SQL best practices, including proper indexing, optimization, and clear formatting with appropriate comments.
                4. Review the generated query and make sure it is correct, executable and will return the data you need.
                5. **IMPORTANT** Finally you will return the sql query using the {function_name} function. You MUST include both the task description and the SQL query in your response. The SQL query should be assigned to the sql_query field of the {function_name} object.
                6. **CRITICAL** Your response MUST be in the following format:
                    {{\n                        "task": "your task description",\n                        "sql_query": "your generated SQL query"\n                    }}\n                Do not include any other fields or text in your response.
        """
        return task_prompt
    
    def get_prompt(self, task, function_name):
        """
        Constructs and returns the full chat prompt template, partially applied with the function name.
        Handles errors in prompt construction.
        """
        try:
            return ChatPromptTemplate.from_messages([
                ("system", self.get_system_prompt()),
                MessagesPlaceholder(variable_name="messages"),
                ("human", self.get_task_prompt(self.schema, self.column_definitions, self.sql_examples, self.guidelines))
            ]).partial(
                #query_execution_tool=query_execution_tool,
                function_name=function_name,
                task=task,
                messages=[],
                schema=self.schema,
                column_definitions=self.column_definitions,
                sql_examples=self.sql_examples,
                guidelines=self.guidelines
            )
        except Exception as e:
            print(f"[ERROR] Failed to construct ChatPromptTemplate: {e}")
            return None


class QueryRevisePrompt():
    """
    Constructs prompts for query revision tasks, loading necessary resources from S3.
    """
    def __init__(self):
        """
        Initializes the QueryRevisePrompt by loading required resources from S3.
        Handles errors during S3 reads and logs them.
        """
        try:
            self.rc_plan = S3_reader()._run('s3://orbit-science-ai-assist/COH-Data/coh_rc_plan.txt')
        except Exception as e:
            print(f"[ERROR] Failed to load rc_plan from S3: {e}")
            self.rc_plan = None
        try:
            self.rc_understanding = S3_reader()._run('s3://orbit-science-ai-assist/COH-Data/Understanding_COH_Root_Cause.txt')
        except Exception as e:
            print(f"[ERROR] Failed to load rc_understanding from S3: {e}")
            self.rc_understanding = None
        try:
            self.column_definitions = S3_reader()._run('s3://orbit-science-ai-assist/COH-Data/coh_v3_column_definition.txt')
        except Exception as e:
            print(f"[ERROR] Failed to load column_definitions from S3: {e}")
            self.column_definitions = None
        try:
            self.schema = S3_reader()._run('s3://orbit-science-ai-assist/COH-Data/schema.txt')
        except Exception as e:
            print(f"[ERROR] Failed to load schema from S3: {e}")
            self.schema = None
        try:
            self.sql_examples = S3_reader()._run('s3://orbit-science-ai-assist/COH-Data/examples.txt')
        except Exception as e:
            print(f"[ERROR] Failed to load sql_examples from S3: {e}")
            self.sql_examples = None
        try:
            self.guidelines = S3_reader()._run('s3://orbit-science-ai-assist/COH-Data/guidance.txt')
        except Exception as e:
            print(f"[ERROR] Failed to load guidelines from S3: {e}")
            self.guidelines = None
        try:
            self.coh_business = S3_reader()._run('s3://orbit-science-ai-assist/COH-Data/guidance.txt')
        except Exception as e:
            print(f"[ERROR] Failed to load coh_business from S3: {e}")
            self.coh_business = None

    def get_system_prompt(self):
        """
        Returns the system prompt for the data analyst agent.
        """
        system_prompt = """
            You are a senior data analyst known for excellent SQL skills and a deep understanding of the AWS Athena database. 
            You are also known for your ability to write clear and concise SQL queries.
            You will write **ANSI SQL** queries to get the data from the AWS Athena database.
        """
        return system_prompt

    def get_task_prompt(self, schema, column_definitions, sql_examples, guidelines):
        """
        Constructs the task prompt using schema, column definitions, SQL examples, and guidelines.
        """
        task_prompt =  """
        You will be given a data collection request as follows and you will generate a SQL query which will be fed into the next tool to get the data.
        <Data Collection Request>
            {task}
        </Data Collection Request>
        Here is the SQL query that you previously generated:
        <Previously Generated SQL Query>
            {sql_query}
        </Previously Generated SQL Query>
        Here is the ERROR message that you received:
        <ERROR Message>
            {error_message}
        </ERROR Message>

        After receiving the data request, previously generated SQL query and the ERROR message, you will:
                1. Revise the SQL query to address the ERROR message. Here are the information and guidelines for you to revise the query:
                    - Use the database schema and column definitions to generate the query
                      -- Here is the database schema:
                        <database schema>
                            {schema}
                        </database schema>
                    -- Here are the column definitions:
                        <column definitions>
                            {column_definitions}
                        </column definitions>
                    -- Here are the examples for your references:
                        <sql examples>
                            {sql_examples}
                        </sql examples>
                    -- Note that the database in the AWS Athena is called "AwsDataCatalog"."ai-assist"."coh__v3_prod"
                    -- Note that the schema and columns listed above are what are currently available in the database.
                2. Follow the guidelines to generate the sql query
                    <guidelines>
                        {guidelines}
                    </guidelines>
                3. All queries will follow SQL best practices, including proper indexing, optimization, and clear formatting with appropriate comments.
                4. Review the generated query and make sure it is correct, executable and will return the data you need.
                5. **IMPORTANT** Finally you will return the sql query using the {function_name} function.
        """
        return task_prompt
    
    def get_prompt(self, task, sql_query, error_message, function_name):
        """
        Constructs and returns the full chat prompt template, partially applied with the function name.
        Handles errors in prompt construction.
        """
        try:
            return ChatPromptTemplate.from_messages([
                ("system", self.get_system_prompt()),
                MessagesPlaceholder(variable_name="messages"),
                ("human", self.get_task_prompt(self.schema, self.column_definitions, self.sql_examples, self.guidelines))
            ]).partial(
                #query_execution_tool=query_execution_tool,
                function_name=function_name,
                task=task,
                sql_query=sql_query,
                error_message=error_message,
                messages=[],
                schema=self.schema,
                column_definitions=self.column_definitions,
                sql_examples=self.sql_examples,
                guidelines=self.guidelines
            )
        except Exception as e:
            print(f"[ERROR] Failed to construct ChatPromptTemplate: {e}")
            return None
