from config import opt
from . import data_process
import os
if not os.path.exists('data/' + opt.dataset+'/train_data.csv') or opt.data_process:
	data_process.process(opt.dataset)
if opt.method == 'PDA':
	data_process.PDA_test_pop(opt.dataset)
