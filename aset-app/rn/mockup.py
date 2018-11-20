import sys

filePath = sys.argv[1]

if __name__ == '__main__':
    with open('{}_icr.txt'.format(filePath), 'w') as f:
        f.write('ICR mock-up')