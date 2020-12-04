import glob, os, shutil

# input: source directory (e.g. tpheno)
prefix = '/Users/pleiades/Documents/work'
prefix_dest = '/Users/pleiades/Dropbox/Project'
moduledir = 'tpheno'
source_dir = os.path.join(prefix, moduledir)
dest_dir = os.path.join(prefix_dest, moduledir)
file_type = 'py'
verbose = True

files = glob.iglob(os.path.join(source_dir, "*.%s" % file_type))
acc = 0
for file in files:
    if os.path.isfile(file):
    	if verbose: 
    		print('> coping %s to %s ...' % (file, dest_dir))
        shutil.copy2(file, dest_dir)
        acc += 1

if verbose: 
	print('> backed up %d files' % acc)

