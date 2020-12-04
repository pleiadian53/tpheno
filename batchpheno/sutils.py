####################################
#
# Shell Utilities 
# 

import commands, shutil
import os

class MemMonitor(object): 
    _proc_status = '/proc/%d/status' % os.getpid()
    _scale = {'kB': 1024.0, 'mB': 1024.0*1024.0,
              'KB': 1024.0, 'MB': 1024.0*1024.0}

    def _VmB(VmKey):
        '''Private.
        '''
        # global _proc_status, _scale
        # get pseudo file  /proc/<pid>/status
        _proc_status, _scale = MemMonitor._proc_status, MemMonitor._scale
        try:
            t = open(_proc_status)
            v = t.read()
            t.close()
        except:
            return 0.0  # non-Linux?
        # get VmKey line e.g. 'VmRSS:  9999  kB\n ...'
        i = v.index(VmKey)
        v = v[i:].split(None, 3)  # whitespace
        if len(v) < 3:
            return 0.0  # invalid format?
        # convert Vm value to bytes
        return float(v[1]) * _scale[v[2]]


    def memory(since=0.0):
        '''Return memory usage in bytes.
        '''
        return MemMonitor._VmB('VmSize:') - since


    def resident(since=0.0):
        '''Return resident memory usage in bytes.
        '''
        return MemMonitor._VmB('VmRSS:') - since

def execute(cmd): 
    """
    
    [note]
    1. running qrymed with this command returns a 'list' of medcodes or None if 
       empty set
    """
    st, output = commands.getstatusoutput(cmd)
    if st != 0:
        raise RuntimeError, "Could not exec %s" % cmd
    #print "[debug] output: %s" % output
    return output  #<format?>

def execute2(cmd): 
    pass 


def run_script(script, stdin=None):
    """
    Run shell script. 
    Returns (stdout, stderr), raises error on non-zero return code

    Memo
    ----
    1. If you don't use stdout=subprocess.PIPE etc then the script will be 
       attached directly to the console. This is really handy if you have, for instance, 
       a password prompt from ssh.
    """
    import subprocess
    # Note: by using a list here (['bash', ...]) you avoid quoting issues, as the 
    # arguments are passed in exactly this order (spaces, quotes, and newlines won't
    # cause problems):
    proc = subprocess.Popen(['bash', '-c', script],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        stdin=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    if proc.returncode:
        raise ScriptException(proc.returncode, stdout, stderr, script)
    return stdout, stderr

class ScriptException(Exception):
    def __init__(self, returncode, stdout, stderr, script):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        Exception.__init__('Error in script')
    def __str__(self):
        print("> Output:\n%s\n> Error:%s" % (self.stdout, self.stderr)) 

def t_cross_ref(): 
    import commands
    script = """
            cd data-diag
            for f in diag_055.9.csv diag_131.09.csv diag_041.01.csv  
            do
                x=$(cat archive/t2/$f | wc -l)   
                y=$(cat $f | wc -l)
                echo "x=$x, y=$y"
            done
          """
    o, e  = run_script(script)
    print("stdout: %s\nstderr: %s\n" % (o, e))
    return

def test(): 
    t_cross_ref() 

if __name__ == "__main__": 
    test()