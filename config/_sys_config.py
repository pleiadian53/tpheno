import ConfigParser
import os
from os import getenv
import sys

config_file = 'tp_system.cfg' # [defaut]
proj_dir = getenv("TP_PROJ_DIR")  # project root dir 
if not proj_dir:
    print('warning> project dir not specified. exiting ...')
    # search parent directory? 
    sys.exit(1)

config_dir = getenv("TP_CONFIG_DIR") # configuration file directory
if not config_dir: config_dir = proj_dir
write_system_noop = True

def write_config(**kargs): 
    """
    Write a default configuration file. 

    Memo
    ----
    1. Parameters are case insensitive under the current settings of ConfigParser 
    """

    global config_file, proj_dir
    if kargs.has_key('config_file'): 
        config_file = kargs['config_file']
    verbose = kargs.get('verbose', False)
    # print('> config dir: %s, config_file: %s' % (config_dir, config_file))
    config_file = os.path.join(config_dir, os.path.basename(config_file))
    
    if write_system_noop and os.path.exists(config_file): 
        if verbose: print('Found configuration at %s.' % config_file)
        return 

    config = ConfigParser.RawConfigParser()

    # When adding sections or items, add them in the reverse order of
    # how you want them to be displayed in the actual file.
    # In addition, please note that using RawConfigParser's and the raw
    # mode of ConfigParser's respective set functions, you can assign
    # non-string values to keys internally, but will receive an error
    # when attempting to write to a file or when you get it in non-raw
    # mode. SafeConfigParser does not allow such assignments to take place.

    sect_name = 'system'
    config.add_section(sect_name)

    if not os.path.exists(proj_dir):
    	curdir = os.getcwd()
    	print('warning> Project directory %s does not exist; use current directory instead: %s' % (proj_dir, curdir))
    	proj_dir = curdir

    config.set(sect_name, 'ProjDir', proj_dir)
    config.set(sect_name, 'ConfigDir', config_dir)

    # directory names under the project (root) directory
    config.set(sect_name, 'Ref', 'ref')  

    # config.set(sect_name, 'a_bool', 'true')
    # config.set(sect_name, 'a_float', '3.1415')

    config.set(sect_name, 'RefDir', '%(ProjDir)s/%(Ref)s')

    config.set(sect_name, 'DataDir', '%(ProjDir)s/data')
    config.set(sect_name, 'DataRoot', '%(ProjDir)s/data')  # source data
    config.set(sect_name, 'DataExpRoot', '%(ProjDir)s/data-exp')  # output and derived data
    config.set(sect_name, 'DataExp', '%(ProjDir)s/data-exp')  # output and derived data
    config.set(sect_name, 'DataIn', '%(ProjDir)s/data-in')  # input data for ML tasks
    config.set(sect_name, 'TestDir', '%(ProjDir)s/test')
    config.set(sect_name, 'LogDir', '%(TestDir)s/log')
    config.set(sect_name, 'Bin', '%(ProjDir)s/bin')  # output and derived data

    # Writing our configuration file to 'example.cfg'
    with open(config_file, 'wb') as configfile:
    	print('info> Writing configurations to %s' % config_file)
        config.write(configfile)

    return

def read_config(**kargs): 
    """
    Read generic configuration settings. 
    """
    global config_file
    if kargs.has_key('config_file'): config_file = kargs['config_file']
    if not os.path.exists(config_file): 
        print('error> config file %s not found.' % config_file)
        sys.exit(1)

    params = {}
    # config = ConfigParser.RawConfigParser()
    config = ConfigParser.SafeConfigParser()
    config.read(config_file)

    # getfloat() raises an exception if the value is not a float
    # getint() and getboolean() also do this for their respective types
    # a_float = config.getfloat('Section1', 'a_float')
    # an_int = config.getint('Section1', 'an_int')

    # Notice that the next output does not interpolate '%(bar)s' or '%(baz)s'.
    # This is because we are using a RawConfigParser().
    # if config.getboolean('Section1', 'a_bool'):
    #     print config.get('Section1', 'foo')
    sect_name = 'system'
    params['ProjDir'] = config.get(sect_name, 'ProjDir', 0)
    params['RefDir'] = config.get(sect_name, 'RefDir', 0)
    params['DataRoot'] = config.get(sect_name, 'DataRoot', 0)

    return params 

def read(param, *args, **kargs): 
    global config_file, proj_dir

    if kargs.has_key('config_file'): config_file = kargs['config_file']
    config_file = os.path.join(config_dir, os.path.basename(config_file))

    if not os.path.exists(config_file): 
        print('error> config file %s not found.' % config_file)
        sys.exit(1)

    # print('info> kargs: %s' % kargs)
    config = ConfigParser.SafeConfigParser(kargs)  # [todo] factor out? 
    config.read(config_file)


    sect_name = kargs.get('section_name', 'system')
    try: 
        return config.get(sect_name, param, 0)
    except Exception, e: 
        print('warning> %s' % e)
        if args:
            return args[0]
    raise ValueError, "No defined value for %s" % param 

