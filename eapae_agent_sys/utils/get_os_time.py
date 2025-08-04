import datetime
def get_timestamp():
    """Returns the current timestamp in a standard format."""
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] 