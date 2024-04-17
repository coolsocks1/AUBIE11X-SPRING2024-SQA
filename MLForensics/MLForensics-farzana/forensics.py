import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def method_with_forensics(input_param):
    try:
        # Your method logic here
        result = perform_operation(input_param)
        logging.info(f"Operation successful with input: {input_param}")
        return result
    except Exception as e:
        logging.error(f"Error in method_with_forensics with input {input_param}: {e}")
        raise  # Re-raise the exception for higher-level handling

def another_method_with_forensics(input_param):
    try:
        # Your method logic here
        result = perform_other_operation(input_param)
        logging.info(f"Another operation successful with input: {input_param}")
        return result
    except Exception as e:
        logging.error(f"Error in another_method_with_forensics with input {input_param}: {e}")
        raise  # Re-raise the exception for higher-level handling

# Define other modified methods with forensics enhancements as needed
# You can include additional methods with similar try-except blocks and logging statements
