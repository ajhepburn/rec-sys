from pathlib import Path

def check_file(path, filename):
    my_file = Path(path+filename)
    if my_file.is_file(): return 1
    else: return 0