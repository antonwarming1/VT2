1) Dont use typescript in the code like def clean_csv(self, df: pd.DataFrame) -> pd.DataFrame:
2) Keep the code simple
3) Dont use functions inside functions like:

def clean_json(self, data: dict) -> dict:
        """
        Clean screwdriver JSON data.
        
        Cleaning steps:
        1. Validate required fields exist
        2. Convert string numbers to proper types
        3. Remove invalid records
        """
        cleaned = data.copy()
        
        # Typical fields in screwdriver data
        numeric_fields = ['torque', 'angle', 'time', 'speed']
        
        def clean_value(value):
            """Convert string numbers and handle None."""
            if value is None or value == '':
                return None
            if isinstance(value, str):
                try:
                    # Try float first
                    return float(value.replace(',', '.'))
                except ValueError:
                    return value
            return value

4) keep the scripts short. If they exceed 500 lines, make new scripts and import the functions. 
