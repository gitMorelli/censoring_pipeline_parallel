from time import perf_counter, time
import logging

class FileWriter:
    def __init__(self, enabled=False,path=None):
        self.enabled = enabled
        self.path=path
        self.time_dict={}
        self.call_start('complete_process',block=True,mode="w")

    
    def call_start(self, key,block=False,mode="a"):
        if block:
            self.write("-"*50,mode=mode)
        self.time_dict[key] = perf_counter()
    
    def call_end(self, key,block=False):
        self.write(f"time for process {key} = {( perf_counter()-self.time_dict[key] ):0.6f}s")
        if block:
            self.write("-"*50)


    def write(self, text: str,mode="a"):
        if not self.enabled:
            return
        content = str(text)
        with open(self.path, mode, encoding="utf-8") as f:
            f.write(content + "\n")

def initialize_logger(verbose,logger):
    if verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.disable(logging.CRITICAL)