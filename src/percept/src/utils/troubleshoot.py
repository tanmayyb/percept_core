import inspect
import traceback

# helper
def get_current_function_name():
    stack = traceback.extract_stack()
    return stack[-3].name 


def get_error_text(e, show_funcname=True, show_error=True, print_stack_trace=True):
    msg = ""
    if show_funcname:  
        msg+= f"error in function: '{get_current_function_name()}':"
    if show_error:  
        msg+= f" {e}"
    if print_stack_trace:
        msg+= f" {traceback.format_exc()}"
    return msg