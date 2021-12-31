import sys
import os
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()


def go():
    #if len(sys.argv) != 1:  
    #    logging.error('Wrong amount of aruments')
    #    sys.exit(1)

    
    logging.info(f"Argument amount: {len(sys.argv)}")
    logging.info(f"Argument argv[0]: {sys.argv[0]}")
    logging.info(f"Argument argv[1]: {sys.argv[1]}")



if __name__ == '__main__':
    go()
        
