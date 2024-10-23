import os
import psutil


def process_info():
    print('=============================>>>  Main  <<<=============================')
    print('parent process id : ', os.getppid())
    print('process id        : ', os.getpid())
    system_memory_info = psutil.virtual_memory()
    init_free_memory = system_memory_info.total - system_memory_info.used
    init_process_memory = psutil.Process(os.getpid()).memory_full_info().uss
    
    return init_free_memory, init_process_memory