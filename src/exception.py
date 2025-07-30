import sys
from src.logger import logging

def error_message_detail(error_object, error_detail:sys):
    # exc_tb will contain all info related to exception like file name, line number, etc
    _,_, exc_tb=error_detail.exc_info()
    
    # Get the file where the exception occurs
    file_name=exc_tb.tb_frame.f_code.co_filename
    
    # Create the custom error message.
    error_message="\n Error occurred in file: [{0}] \n Line number: [{1}] \n Error Message: [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error_object)
    )

    return error_message

class CustomException(Exception):
    def __init__(self, error_object, error_detail:sys):
        super().__init__(error_object)
        self.error_message=error_message_detail(error_object, error_detail=error_detail)

    def __str__(self):
        return self.error_message
