from Config import Config
from train_and_test import train_and_test


train_and_test(Config("../../personality_dataset", True, "distance", -1, True))