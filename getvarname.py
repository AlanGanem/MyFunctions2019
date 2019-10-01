import inspect


def get_var_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [k for k, v in callers_local_vars if v is var]
