import os, glob
from config import config

freqs = ['alpha', 'beta', 'delta', 'theta', 'lgamma', 'hgamma', 'broadband', 'raw']

# for freq in freqs:
#     print(len(glob.glob(os.path.join(config['resultsdir'], freq + '_recon', '*'))))

def prune(freq):
    files = glob.glob(os.path.join(config['resultsdir'], freq + '_recon/*.npz'))
    m = dict()
    files = [os.path.splitext(x)[0] for x in files]
    for f in files:
        if 'within' in f:
            f = f.split('_within')[0]
            if f in m:
                m[f] += 1
            else:
                m[f] = 1
        else:
            if f in m:
                m[f] += 1
            else:
                m[f] = 1
    pruned = []
    for k in m.keys():
        if m[k] == 2:
            pruned.append(k)
        # else:
        #     os.rename(k+'.npz', os.path.join(config['resultsdir'], 'orphans', os.path.basename(k) + '.npz'))
    
    print(freq + ' before: ' + str(len(m)))
    print('after: ' + str(len(pruned)))
    return set([x+'.npz' for x in pruned])

for freq in freqs:
    prune(freq)

# s = set([''.join(os.path.basename(x).split('_alpha')[::2]) for x in list(prune('alpha'))])
# for freq in freqs:
#     t = set([''.join(os.path.basename(x).split('_' + freq)[::2]) for x in list(prune(freq))])
#     print(s==t)

# errs = glob.glob('/dartfs-hpc/rc/home/4/f003f64/recon.e*')
# files = []
# for err in errs:
#         f = open(err)
#         txt= f.read()
#         f.close()
#         try:
#             split = txt.split('RAM_union/')[1].split('``')[0]
#             files.append(split)
#         except:
#             pass

# print(len(files))

# def model_set(freq):
#     files = glob.glob(os.path.join(config['modeldir'], '*' + freq + '*'))
#     assert len(files) == 79
#     files = [os.path.basename(x).split('_' + freq)[0] for x in files]
#     return set(files)

# se = model_set('alpha')
# for freq in freqs:
#     print(freq)
#     print(se == model_set(freq))

# for f in glob.glob(os.path.join(config['resultsdir'], '*/*ORPHAN*')):
#     newf = os.path.join(config['resultsdir'],'orphans',os.path.basename(f))
#     os.rename(f, newf)