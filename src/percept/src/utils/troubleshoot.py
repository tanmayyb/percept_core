import inspect
import traceback

# helper
def get_current_function_name():
    stack = traceback.extract_stack()
    return stack[-2].name 


def get_error_text(e, show_funcname=True, show_error=True):
    msg = ""
    if show_funcname:  msg+= f"error in function: {get_current_function_name}"
    if show_error:  msg+= f"{e}"
    return msg