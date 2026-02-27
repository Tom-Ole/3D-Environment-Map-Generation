import sys
from args import parse_args
from pprint import pp

def main() -> None:
    
    args = parse_args()
    
    pp(args)
    
    return
    
if __name__ == "__main__":
    sys.exit(main())