import enum
import time

class ModeStatus(enum.Enum):
    TRAIN = 1
    VAL = 2
    TEST = 3
    PREDICT = 4

def LOG(INFO=None):
    basic_str_begin = "%%%%%%%%%%%%%%%%"
    basic_str_end = "================"
    print(basic_str_begin)
    if  INFO:
        print(INFO)
    print(basic_str_end)


def write_loss(loss, log_path="./log/"):
    t = str(int(time.time()))
    file_name = log_path + t
    with open(file_name, 'w') as f:
        for i in range(len(loss)):
            f.write(str(loss[i]))
            f.write('\n')
