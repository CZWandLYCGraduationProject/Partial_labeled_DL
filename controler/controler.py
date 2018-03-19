import tensorflow as tf
from utils import *
import json
from data_handler.data_handler import DataHandler
from dawn.dawn import DawnArchitect
import time

train_mode = True
def main(config_path):
    '''
    This function is the main controller of the system.
    First loading the configs
    :param config_path:
    :return: None
    '''
    morning = initialize(config_path)
    LOG("Initialized and enter the main program")
    if train_mode:
        step = morning.get_setp()
        LOG("step is {0}".format(step))
        loss = 0
        loss_list = []
        start_time = time.time()
        for i in range(50000):
            loss += morning.train_a_step()
            step += 1
            if step != 0 and step % 100 == 0:
                print("STEP:{0}, AVGLOSS:{1}".format(step, loss / 100))
                loss_list.append(loss)
                loss = 0
            # loss = morning.train_a_step()
            # loss_list.append(loss)
            if step !=0 and step % 5000 == 0:
                morning.save()
            # # if step != 0 and step % 500 == 0:
            # #     morning.val()
            # if step != 0 and step % 100 == 0:
            #     print(loss)
        end_time = time.time()
        LOG("Total Time is {0}".format(end_time - start_time))
        write_loss(loss_list)
        time.sleep(10)
    else:
        morning.predict()

def initialize(config_path):
    global train_mode
    config = open(config_path, "r")
    config = json.load(config)
    LOG("Load Config")
    morning = DawnArchitect(config)
    train_mode = config["train"]
    print("INFO: Initialized!")
    return morning

if __name__ == "__main__":
    main("./configs/training_configs.json")
