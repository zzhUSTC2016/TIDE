def print_str(file_path, str_to_print, file=True, window=True):
    """
    To print string to log file and the window
    """
    if file:
        with open(file_path, 'a+') as f_log:
            f_log.write(str_to_print)
            f_log.write('\n')
    if window:
        print(str_to_print)