from scripts import create_splits
from datetime import datetime
import sys

if __name__=="__main__":
    scriptStart = datetime.now()
    
    splitspath = str(sys.argv[1])
    traindir = str(sys.argv[2])
    testdir = str(sys.argv[3])
    imgloc = str(sys.argv[4])
    
    create_splits.reorganize(traindir, splitspath, imgloc)
    create_splits.reorganize(testdir, splitspath, imgloc, split_type='Test')
    print('Total script time: ', datetime.now()-scriptStart)