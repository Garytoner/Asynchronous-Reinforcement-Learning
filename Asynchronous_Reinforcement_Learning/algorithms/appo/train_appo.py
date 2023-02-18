import sys
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

def main():
    """Left here for legacy reasons. Use the Asynchronous_Reinforcement_Learning.run_algorithm script from the root folder instead."""
    from Asynchronous_Reinforcement_Learning.run_algorithm import main as run_algorithm_main
    return run_algorithm_main()


if __name__ == '__main__':
    sys.exit(main())
