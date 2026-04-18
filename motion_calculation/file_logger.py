import os
from datetime import datetime

class FileLogger():
    def __init__(self):
        self.text_file = None
        self.write_to_file = False
    
    def set_folder_and_file(self, folder_path, log_file_name) -> None:
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        file_path = folder_path + "/" + log_file_name
        self.text_file = open(file_path, "a")
        self.write_to_file = True
    
    def __log_to_file(self, log_string: str):
        if not self.write_to_file:
            return
        if self.text_file == None:
            raise Exception("FileLogger: folder and file not set")
        self.text_file.write(log_string + "\n")
        self.text_file.flush()

    def info(self, log_string):
        string_with_time = "[" + datetime.today().isoformat() + "]: " + log_string
        print(string_with_time)
        self.__log_to_file(string_with_time)

if __name__ == "__main__":
    # Test Class
    fl = FileLogger("logs/test", "test_log.txt")

    for i in range(10):
        fl.info(f"This a test string {i}", True)
