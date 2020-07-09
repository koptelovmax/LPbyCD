import numpy as np
import sys

name = sys.argv[1] # 'Enzyme', 'GPCR', 'IC', 'NR' or 'Kinase'

# number of folds:
k = 5

work_path = 'data//internal_cv'
#%%
def load_base_networks():
    
    lig = []
    tar = []
    
    # Load targets similarities (computed by SmithWaterman score):
    f = open("data//"+name+"//target-sim_smiwat.txt", 'r')
    
    for line in f:
        try:
            data = line.split(' ')
            tar.append(data[0])
            tar.append(data[1])
        except ValueError:
            print "Invalid input:", line
                    
    f.close()
    
    print '1 out of 3 networks loaded'
    
    # Load ligands similarities (computed by Simcomp score):
    f = open("data//"+name+"//drug-sim_simcomp.txt", 'r')
    
    for line in f:
        try:
            data = line.split(' ')
            lig.append(data[0])
            lig.append(data[1])
        except ValueError:
            print "Invalid input:", line
                    
    f.close()
    
    print '2 out of 3 networks loaded'
    
    # Load ligands-target interaction network (KEGG):
    f = open("data//"+name+"//drug-target_kegg.txt", 'r')
    
    for line in f:
        try:
            data = line.split(' ')
            lig.append(data[0])
            tar.append(data[1])
        except ValueError:
            print "Invalid input:", line
                    
    f.close()
    
    print '3 out of 3 networks loaded'    
    print 'Graph has been loaded!'
    
    return list(np.unique(lig)),list(np.unique(tar))
#%%
# Load data:
ligands,targets = load_base_networks()

# Separate positive and negative links:
links = []

# Load ligand-target interactions:
f = open("data//"+name+"//drug-target_kegg.txt", 'r')

count = 0
for line in f:
    try:
        data = line.split(' ')
        if data[2].rstrip() != '0':
            links.append(data[0]+' '+data[1])
            count+=1
    except ValueError:
        print "Invalid input:", line

f.close()
#%%
# determine number of ligands and targets:
n = len(targets) # number of targets
m = len(ligands) # number of ligands

total_num = n*m
total_num_pos = (len(links))
total_num_neg = total_num-len(links)

# list of all possible links:
dt_links_all = np.zeros((total_num,2),np.int)
dt_links_pos = np.zeros((total_num_pos,2),np.int)
dt_links_neg = np.zeros((total_num_neg,2),np.int)

count = 0
count_pos = 0
count_neg = 0
for t in range(n):
    for l in range(m):
        dt_links_all[count,0] = np.int(ligands[l][1:])
        dt_links_all[count,1] = np.int(targets[t][1:])
        if ligands[l]+' '+targets[t] in links:
            dt_links_pos[count_pos,0] = np.int(ligands[l][1:])
            dt_links_pos[count_pos,1] = np.int(targets[t][1:])
            count_pos+=1
        else:
            dt_links_neg[count_neg,0] = np.int(ligands[l][1:])
            dt_links_neg[count_neg,1] = np.int(targets[t][1:])
            count_neg+=1
        count+=1
#%%
# Split positive links into 5 folds
        
# determine sizes of positive folds:
folds_pos = []
for j in range(k-1):
    folds_pos.append(total_num_pos/k)
folds_pos.append(total_num_pos/k + total_num_pos % k)

# Split negative links into 5 folds

# determine sizes of negative folds:
folds_neg = []
for j in range(k-1):
    folds_neg.append(total_num_neg/k)
folds_neg.append(total_num_neg/k + total_num_neg % k)

# permute indexes of positive and negative links:
test_indexes_pos = np.random.permutation(total_num_pos)
test_indexes_neg = np.random.permutation(total_num_neg)
#%%
#fold_start = 0
fold_pos_start = 0
fold_neg_start = 0

for j in range(k):
    
    # original file with all links:
    f = open("data//"+name+"//drug-target_kegg.txt", 'r')
    
    # output files:
    f_out_train = open(work_path+"//"+name+"//drug-target_kegg_train_"+str(j+1)+".txt", 'w')
    
    fold_nodes = []
        
    fold_pos_indexes = test_indexes_pos[fold_pos_start:fold_pos_start+folds_pos[j]]
    for l in fold_pos_indexes:
        fold_nodes.append((dt_links_pos[l,0],dt_links_pos[l,1]))
            
    fold_neg_indexes = test_indexes_neg[fold_neg_start:fold_neg_start+folds_neg[j]]
    for l in fold_neg_indexes:
        fold_nodes.append((dt_links_neg[l,0],dt_links_neg[l,1]))
        
    np.save(work_path+"//"+name+"//test_nodes_"+str(j+1)+".npy",np.array(fold_nodes))
    
    count = 0
    for line in f:
        try:
            if count not in fold_pos_indexes:
                f_out_train.write(line)
            count+=1
        except ValueError:
            print "Invalid input:", line
    f.close()
      
    f_out_train.close()
    
    fold_pos_start+=folds_pos[j]
    fold_neg_start+=folds_neg[j]
#%%