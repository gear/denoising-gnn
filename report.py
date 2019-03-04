import glob
import numpy as np
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Produce report from result files")
    parser.add_argument('--path', type=str, default="",
                        help="Path to the result files (* will be appended)")
    args = parser.parse_args()
    
    test_accuracies = []
    train_accuracies = []
    train_losses = []
    
    for f in glob.glob(args.path+"*"):
        with open(f) as ff:
            loss, train_acc, test_acc = map(float, ff.readline().split())
            test_accuracies.append(test_acc)
            train_accuracies.append(train_acc)
            train_losses.append(loss) 
     
    print("Test: {:.4f} ± {:.4f}".format(np.mean(test_accuracies),\
                                         np.std(test_accuracies))) 
    print("Train: {:.4f} ± {:.4f}".format(np.mean(train_accuracies),\
                                          np.std(train_accuracies))) 
    print("Loss: {:.4f} ± {:.4f}".format(np.mean(loss),\
                                         np.std(loss))) 

if __name__ == "__main__":
    main()
