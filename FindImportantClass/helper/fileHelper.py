# File Related
def read_node_label(filename, skip_head=False):
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        if skip_head:
            fin.readline()
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        X.append(vec[0])
        Y.append(vec[1])
        print(vec[0])
    fin.close()
    return X, Y

def save_node_label(filename, x,y,z):
    f_out = open(filename, 'w+')
    i = 0
    for idx in x:
        t=''
        t = str(x[i]) + ' '+str(y[i]) + ' '+str(z[i])
        f_out.write(t)
        f_out.write('\n')
        i = i + 1
    f_out.close()
    return x,y,z

def save_rank(filename, x):
    f_out = open(filename, 'w+')
    i = 0
    for idx in x:
        t=''
        t = str(x[i])
        f_out.write(t)
        f_out.write('\n')
        i = i + 1
    f_out.close()