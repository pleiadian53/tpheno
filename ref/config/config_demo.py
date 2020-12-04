import ConfigParser
import os

config_file = 'example.cfg'

def write_config(): 
    config = ConfigParser.RawConfigParser()

    config.add_section('Section1')
    config.set('Section1', 'an_int', '15')
    config.set('Section1', 'a_bool', 'true')
    config.set('Section1', 'a_float', '3.1415')
    config.set('Section1', 'baz', 'fun')
    config.set('Section1', 'bar', 'Python')
    config.set('Section1', 'foo', '%(bar)s is %(baz)s!')

    # Writing our configuration file to 'example.cfg'
    with open(config_file, 'wb') as configfile:
        config.write(configfile)

def read_raw_config(): 
    config = ConfigParser.RawConfigParser()
    config.read(config_file)

    # getfloat() raises an exception if the value is not a float
    # getint() and getboolean() also do this for their respective types
    a_float = config.getfloat('Section1', 'a_float')
    an_int = config.getint('Section1', 'an_int')
    print a_float + an_int

    # Notice that the next output does not interpolate '%(bar)s' or '%(baz)s'.
    # This is because we are using a RawConfigParser().
    if config.getboolean('Section1', 'a_bool'):
        print config.get('Section1', 'foo')

    return

def read_config(): 
    config = ConfigParser.ConfigParser()
    config.read('example.cfg')

    # Set the third, optional argument of get to 1 if you wish to use raw mode.
    print config.get('Section1', 'foo', 0)  # -> "Python is fun!"
    print config.get('Section1', 'foo', 1)  # -> "%(bar)s is %(baz)s!"

    # The optional fourth argument is a dict with members that will take
    # precedence in interpolation.
    print config.get('Section1', 'foo', 0, {'bar': 'Documentation',
                                        'baz': 'evil'})
    return

def test(**kargs): 
    if not os.path.exists(config_file):
    	print('> writing a new configuration file: %s' % config_file)
        write_config() 
    read_raw_config()

if __name__ == "__main__": 
    test()


