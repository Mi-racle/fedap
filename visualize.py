from pathlib import Path

from libs import log_round
from logger import Logger
import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logger = Logger(Path('fednoniidprox30'))

fin = open('/home/miracle/fedyolo/global_map_noniid_prox30.txt', 'r')
maps = fin.read().split()
maps = [float(m) for m in maps]
fin.close()

for i, m in enumerate(maps):
    log_round(logger, i, 'mAP', m)
