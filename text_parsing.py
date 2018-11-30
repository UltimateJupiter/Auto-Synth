import numpy as np

def ret_parse():
    fl = open("./testing_params.txt")
    
    queue = list()
    ls = [n for n in fl]
    assert len(ls) == 147, len(ls)

    for line in ls:

        tmp = line.split("|")
        ret = int(float(tmp[1]))
        queue.append(ret)
    
    arr = np.asarray(queue) 
    # arr = arr/100
    return arr
