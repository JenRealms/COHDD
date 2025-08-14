from utils.S3_reader import S3_reader

class coh_doc():
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

    def get_rc_plan(self):
        return self.rc_plan

    def get_rc_understanding(self):
        return self.rc_understanding

    def get_column_definitions(self):
        return self.column_definitions
    
    def get_coh_business(self):
        return self.coh_business
    
    def get_schema(self):
        return self.schema
    
    def get_sql_examples(self):
        return self.sql_examples
    
    def get_guidelines(self):
        return self.guidelines
    
    def get_all_docs(self):
        return {
            "rc_plan": self.get_rc_plan(),
            "rc_understanding": self.get_rc_understanding(),
            "column_definitions": self.get_column_definitions(),
            "coh_business": self.get_coh_business(),
            "schema": self.get_schema(),
            "sql_examples": self.get_sql_examples(),
            "guidelines": self.get_guidelines()
        }