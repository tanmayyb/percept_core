import inspect
import traceback

# helper
def get_current_function_name():
    stack = traceback.extract_stack()
    return stack[-3].name 


def get_error_text(e, show_funcname=True, show_error=False, print_stack_trace=False):
    msg = ""
    if show_funcname:  
        msg+= f"error in function: '{get_current_function_name()}':"
    if show_error:  
        msg+= f" {e}"
    msg += f" {str(e)[:300]}...{str(e)[-450:]}"  # Only show first 50 chars of error
    if print_stack_trace:
        msg+= f" {traceback.format_exc()}"
    return msg