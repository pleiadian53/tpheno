import ConfigParser
import os
from os import getenv
import sys
import _sys_config as sysconfig

sysconfig.write_config(config_file='tp_system.cfg') # if the .cfg file exists then no-op

def read(param, *args, **kargs): 
    return sysconfig.read(param, *args, **kargs)

def test(): 
    def show(adict):
        for k, v in adict.items(): 
            print('%s -> %s' % (k, v))
        return

    # if not os.path.exists(config_file): 
    sysconfig.write_config()
    params = sysconfig.read_config()
    show(params)

    print('dataroot? %s' % read('DataRoot'))
    print('testdir? %s' % read('TestDir', os.getcwd()))

    return


if __name__ == "__main__": 
	test()