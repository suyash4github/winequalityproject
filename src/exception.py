import sys
from src.logger import logging


def error_detail_message(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error occured in python script [{0}] line no [{1}] error message [{2}]".format(file_name,
        exc_tb.tb_lineno,str(error))
    return error_message

class customexception(Exception):
    def __init__(self, error_message,error_detail:sys):
        super().__init__(error_message)    
        self.error_message= error_detail_message(error_message,error_detail=error_detail)

    def __str__(self) -> str:
        return self.error_message
    


