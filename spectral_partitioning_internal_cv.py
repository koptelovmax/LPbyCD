#%%
import numpy as np
import networkx as nx
import timeit
import sys

from sklearn import metrics

name = sys.argv[1] # 'Enzyme', 'GPCR', 'IC', 'NR' or 'Kinase'

matching_setting = sys.argv[2] # 'com-com' or 'node-com'

fold = sys.argv[3] # 1 to 5 folds of internal CV

thresholds = ['default','global mean','global median','localized mean','localized median','localized sum','individualized mean','individualized median','individualized sum']

max_m = 25

work_path = 'results//spectral_partitioning_internal_cv//'+matching_setting+'//'+fold
#%%
def load_base_networks_train():
    
    G = nx.MultiGraph()
    
    lig = []
    tar = []
    networks = {}
    
    net = []
    count = 0
    
    # Load targets similarities (computed by SmithWaterman score):
    f = open("data//"+name+"//target-sim_smiwat.txt", 'r')
    
    net = []
    count = 0
    for line in f:
        try:
            data = line.split(' ')
            G.add_edge(data[0],data[1],weight=float(data[2]),key='tar_sw',label='tar_sw') # target-target similarity
            tar.append(data[0])
            tar.append(data[1])
            net.append(data[0])
            net.append(data[1])
            count+=1
        except ValueError:
            print "Invalid input:", line
            
    networks['tar_sw'] = np.unique(net)
        
    f.close()
    
    print '1 out of 3 networks loaded'
    
    # Load ligands similarities (computed by Simcomp score):
    f = open("data//"+name+"//drug-sim_simcomp.txt", 'r')
    
    net = []
    count = 0
    for line in f:
        try:
            data = line.split(' ')
            G.add_edge(data[0],data[1],weight=float(data[2]),key='lig_sc',label='lig_sc') # ligand-ligand similarity
            lig.append(data[0])
            lig.append(data[1])
            net.append(data[0])
            net.append(data[1])
            count+=1
        except ValueError:
            print "Invalid input:", line
            
    networks['lig_sc'] = np.unique(net)
        
    f.close()
    
    print '2 out of 3 networks loaded'
    
    # Load ligand-target interaction network (KEGG):
    f = open("data//internal_cv//"+name+"//drug-target_kegg_train_"+str(fold)+".txt", 'r')
    
    net = []
    count = 0
    for line in f:
        try:
            data = line.split(' ')
            G.add_edge(data[0],data[1],weight=float(data[2]),key='kegg',label='kegg') # ligand-ligand similarity
            lig.append(data[0])
            tar.append(data[1])
            net.append(data[0])
            net.append(data[1])
            count+=1
        except ValueError:
            print "Invalid input:", line
            
    networks['kegg'] = np.unique(net)
        
    f.close()
    
    # Load rest of nodes:
    f = open("data//"+name+"//drug-target_kegg.txt", 'r')
    
    net = []
    count = 0
    for line in f:
        try:
            data = line.split(' ')
            lig.append(data[0])
            tar.append(data[1])
            net.append(data[0])
            net.append(data[1])
            count+=1
        except ValueError:
            print "Invalid input:", line
            
    networks['kegg'] = np.unique(net)
        
    f.close()
    
    print '3 out of 3 networks loaded'
    
    print 'Graph has been loaded!\n'
    
    return G,list(np.unique(lig)),list(np.unique(tar)),networks
#%%
def load_base_networks():
    
    G = nx.MultiGraph()
    
    lig = []
    tar = []
    networks = {}
    
    net = []
    count = 0
    
    # Load targets similarities (computed by SmithWaterman score):
    f = open("data//"+name+"//target-sim_smiwat.txt", 'r')
    
    net = []
    count = 0
    for line in f:
        try:
            data = line.split(' ')
            G.add_edge(data[0],data[1],weight=float(data[2]),key='tar_sw',label='tar_sw') # target-target similarity
            tar.append(data[0])
            tar.append(data[1])
            net.append(data[0])
            net.append(data[1])
            count+=1
        except ValueError:
            print "Invalid input:", line
            
    networks['tar_sw'] = np.unique(net)
        
    f.close()
    
    print '1 out of 3 networks loaded'
    
    # Load ligands similarities (computed by Simcomp score):
    f = open("data//"+name+"//drug-sim_simcomp.txt", 'r')
    
    net = []
    count = 0
    for line in f:
        try:
            data = line.split(' ')
            G.add_edge(data[0],data[1],weight=float(data[2]),key='lig_sc',label='lig_sc') # ligand-ligand similarity
            lig.append(data[0])
            lig.append(data[1])
            net.append(data[0])
            net.append(data[1])
            count+=1
        except ValueError:
            print "Invalid input:", line
            
    networks['lig_sc'] = np.unique(net)
        
    f.close()
    
    print '2 out of 3 networks loaded'
    
    # Load ligand-target interaction network (KEGG):
    f = open("data//"+name+"//drug-target_kegg.txt", 'r')
    
    net = []
    count = 0
    for line in f:
        try:
            data = line.split(' ')
            G.add_edge(data[0],data[1],weight=float(data[2]),key='kegg',label='kegg') # ligand-ligand similarity
            lig.append(data[0])
            tar.append(data[1])
            net.append(data[0])
            net.append(data[1])
            count+=1
        except ValueError:
            print "Invalid input:", line
            
    networks['kegg'] = np.unique(net)
        
    f.close()
    
    print '3 out of 3 networks loaded'
    
    print 'Graph has been loaded!\n'
    
    return G,list(np.unique(lig)),list(np.unique(tar)),networks
#%%
def partitioning(G,m,threshold):   
    # Laplacian matrix:
    L = nx.laplacian_matrix(G).todense()
    
    # eigen values and eigen vectors:
    w,v = np.linalg.eig(L)
    
    # sort eigen vectors in increasing order of eigen values
    idx = np.argsort(w)
    w = w[idx]
    v = v[:,idx]
    
    # threshold to detect communities:
    if threshold == 'default':
        # default threshold:
        tr = 0
        # extract eigen vectors w.r.t. m:
        v = v[:,1:m+1]
    elif threshold == 'global mean':
        # global mean threshold:
        tr = np.mean(v)
        # extract eigen vectors w.r.t. m:
        v = v[:,1:m+1]
    elif threshold == 'global median':
        # global median threshold:
        tr = np.median(np.array(v))
        # extract eigen vectors w.r.t. m:
        v = v[:,1:m+1]
    elif threshold == 'localized mean':
        # extract eigen vectors w.r.t. m:
        v = v[:,1:m+1]
        # localized mean threshold:
        tr = np.mean(v)
    elif threshold == 'localized median':
        # extract eigen vectors w.r.t. m:
        v = v[:,1:m+1]
        # localized median threshold:
        tr = np.median(np.array(v))
    elif threshold == 'localized sum':
        # extract eigen vectors w.r.t. m:
        v = v[:,1:m+1]
        # localized sum threshold:
        tr = np.sum(v)
    elif threshold == 'individualized mean':
        # extract eigen vectors w.r.t. m:
        v = v[:,1:m+1]
        # individualized mean threshold:
        tr = [np.mean(v[:,i]) for i in range(m)]
    elif threshold == 'individualized median':
        # extract eigen vectors w.r.t. m:
        v = v[:,1:m+1]
        # individualized median threshold:
        tr = [np.median(np.array(v[:,i])) for i in range(m)]
    elif threshold == 'individualized sum':
        # extract eigen vectors w.r.t. m:
        v = v[:,1:m+1]
        # individualized sum threshold:
        tr = [np.sum(v[:,i]) for i in range(m)]
    
    # check threshold:
    v = v > tr
    
    # compute groups:
    return [np.int(''.join([str(np.int(val[0,i])) for i in range(m)]),2) for val in v]
#%%
# Evaluate different threshold settings and m values:
evaluation_results = {}

setting_count = 0
for threshold_setting in thresholds:
    
    for num_eigen_values in range(1,max_m+1):
                
        # Load data:
        G,ligands,targets,layers = load_base_networks_train()
               
        n = len(targets) # number of targets
        m = len(ligands) # number of ligands
        
        f_out = open(work_path+'//log//'+name+'_'+threshold_setting+'_'+str(num_eigen_values)+'_statistics.txt','w')
        
        print 'Some statistics:'
        print 'Nodes: ',G.number_of_nodes()
        f_out.write('Nodes: '+str(G.number_of_nodes())+'\n')
        print 'Edges: ',G.number_of_edges()
        f_out.write('Edges: '+str(G.number_of_edges())+'\n')
        print 'Connected components: ',nx.number_connected_components(G)
        f_out.write('Connected components: '+str(nx.number_connected_components(G))+'\n')
        print 'Ligands: ',m
        f_out.write('Ligands: '+str(m)+'\n')
        print 'Targets:',n
        f_out.write('Targets: '+str(n)+'\n')
        print 'Control sum: ',n+m,'\n'
        f_out.write('Control sum: '+str(n+m)+'\n')
        
        f_out.close()
        
        # k x k cross-validation (CV):
        k = 5 # 5x5 CV
        dt_network = 'kegg' # drug-target network name for experiments
        
        total_num = n*m # total number of all possible links
        dt_links_all = np.zeros((total_num,2),np.int)
        
        # list of all possible dt-links (existing and none-existing):
        count = 0
        for t in range(n):
            for l in range(m):
                dt_links_all[count,0] = l
                dt_links_all[count,1] = t
                count+=1
                
        # determine sizes of folds:
        folds = []
        for j in range(k-1):
            folds.append(total_num/k)
        folds.append(total_num/k + total_num % k)   
        
        f_out = open(work_path+'//log//'+name+'_'+threshold_setting+'_'+str(num_eigen_values)+'.txt','w')
        
        # evaluation scores initialization:
        aucs = []
        auprs = []
        
        for h in range(k): # repeat process k times
        
            # randomly distribute link indexes into folds:
            test_indexes = np.random.permutation(total_num)
               
            fold_start = 0
            for j in range(k):
                interacting_links = [] # list of all interacting dt-links 
                
                # Inside a test fold remove all existing (interacting) links:
                fold_indexes = test_indexes[fold_start:fold_start+folds[j]]
                fold_start+=folds[j]
                
                removed_links = [] # list of removed links
                fold_links = [] # list of links in the fold
                       
                # Iterate over links in a fold:
                for i in fold_indexes:
                    dt_drug = 'l'+str(dt_links_all[i,0])
                    dt_target = 't'+str(dt_links_all[i,1])
                    fold_links.append((dt_drug,dt_target))
                    
                    # Check if link exists (one dt-layer is supported at the moment only):
                    if G.has_edge(dt_drug,dt_target):
                        
                        # memorise link and its class:
                        removed_links.append((dt_drug,dt_target,G.get_edge_data(dt_drug,dt_target)[dt_network]['weight']))
                        
                        if G.get_edge_data(dt_drug,dt_target)[dt_network]['weight'] == 1:
                            interacting_links.append((dt_drug,dt_target))
                        
                        # remove it from G:
                        G.remove_edge(dt_drug,dt_target)
                        
                print 'Prediction on fold #',j+1,'experiment #',h+1,'threshold=',threshold_setting,'m=',num_eigen_values
                f_out.write('Prediction on fold #'+str(j+1)+' experiment #'+str(h+1)+' threshold='+threshold_setting+' m='+str(num_eigen_values)+'\n\n')
                
                start = timeit.default_timer() # Initialize timer to compute time
        
                # Run partitioning:
                groups = partitioning(G,num_eigen_values,threshold_setting)
                
                # Some statistics:
                groups_unique,groups_sizes = np.unique(groups,return_counts=True)
                
                print 'Maximum number of communities: ',2**num_eigen_values
                f_out.write('Maximum number of communities: '+str(2**num_eigen_values)+'\n')
                print 'Number of discovered communities: ',len(groups_unique)
                f_out.write('Number of discovered communities: '+str(len(groups_unique))+'\n')
                print 'Communities: ',groups_unique
                f_out.write('Communities: '+str(groups_unique)+'\n')
                print 'Communities sizes: ',groups_sizes
                f_out.write('Communities sizes: '+str(groups_sizes)+'\n\n')
                       
                # Dictionary of communities:
                communities = {}
                
                for group in groups_unique:
                    communities[group] = []
                
                for i in range(len(G.nodes())):
                    communities[groups[i]].append(G.nodes()[i])
                        
                # Separate communities by type (ligand, target or both):
                communities_lig = [group for group in communities.keys() if any(['l' in node for node in communities[group]])]
                communities_tar = [group for group in communities.keys() if any(['t' in node for node in communities[group]])]
                
                # Dictionary of link probabilities:
                links_probs = {}
                
                # Obtain real classes:
                Y_test = [(couple in interacting_links) for couple in fold_links]
                
                # Obtain link probabilities:
                if matching_setting == 'com-com':
                    
                    # Match communities each with each and compute probabilities of none existing links in each combination:
                    for comm_l in communities_lig:
                        for comm_t in communities_tar:
                            # Number of links between them:
                            comm_links_exist = 0
                            for node_l in communities[comm_l]:
                                for node_t in communities[comm_t]:
                                    if ('l' in node_l) and ('t' in node_t) and G.has_edge(node_l,node_t):
                                        comm_links_exist+=1
                            # Maximum possible number of links:
                            comm_links_max = np.sum(['l' in el for el in communities[comm_l]])*np.sum(['t' in el for el in communities[comm_t]])
                            # Link probability and vector of communities probabilities update:
                            link_prob = comm_links_exist/float(comm_links_max)
                            for node_l in communities[comm_l]:
                                for node_t in communities[comm_t]:
                                    if ('l' in node_l) and ('t' in node_t):
                                        links_probs[node_l,node_t] = link_prob
                    
                    # Link probabilities:                    
                    Y_scores = [links_probs[couple] for couple in fold_links]
                    
                elif 'node-com' in matching_setting:
                    
                    # Match ligands with target communities and compute probabilities of none existing links in each combination:
                    for ligand in ligands:
                        for comm_t in communities_tar:
                            # Number of links between them:
                            comm_links_exist = 0
                            for node_t in communities[comm_t]:
                                if ('t' in node_t) and G.has_edge(ligand,node_t):
                                    comm_links_exist+=1
                            # Maximum possible number of links:
                            comm_links_max = np.sum(['t' in el for el in communities[comm_t]])
                            # Link probability and vector of communities probabilities update:
                            link_prob = comm_links_exist/float(comm_links_max)
                            for node_t in communities[comm_t]:
                                if ('t' in node_t):
                                    links_probs[ligand,node_t] = [link_prob]
                    
                    # Match targets with ligand communities and compute probabilities of none existing links in each combination:
                    for target in targets:
                        for comm_l in communities_lig:
                            # Number of links between them:
                            comm_links_exist = 0
                            for node_l in communities[comm_l]:
                                if ('l' in node_l) and G.has_edge(node_l,target):
                                    comm_links_exist+=1
                            # Maximum possible number of links:
                            comm_links_max = np.sum(['l' in el for el in communities[comm_l]])
                            # Link probability and vector of communities probabilities update:
                            link_prob = comm_links_exist/float(comm_links_max)
                            for node_l in communities[comm_l]:
                                if ('l' in node_l):
                                    links_probs[node_l,target].append(link_prob)
                    
                    # Take an average between two local models:
                    Y_scores = [np.mean(links_probs[couple]) for couple in fold_links]
        
                stop = timeit.default_timer() # stop time counting
                print 'Iteration time, sec: '+str(stop-start)+'\n'
                f_out.write('Time spent, sec: '+str(stop-start)+'\n\n')
                
                # Put removed links back to G (w.r.t. memorised classes):
                for e in removed_links:
                    G.add_edge(e[0],e[1],weight=e[2],key=dt_network,label=dt_network)
                
                # Evaluation by ROC-curve:
                fpr,tpr,_ = metrics.roc_curve(Y_test,Y_scores)
                aucs.append(metrics.auc(fpr,tpr)) # AUC score
                
                # AUPR-curve:
                precision,recall,_ = metrics.precision_recall_curve(Y_test,Y_scores) # PR-curve
                auprs.append(metrics.auc(recall,precision)) # AUPR score
        
        f_out.close()
        
        # memorize results:      
        evaluation_results[setting_count] = {}
        evaluation_results[setting_count]['threshold'] = threshold_setting
        evaluation_results[setting_count]['m'] = num_eigen_values
        evaluation_results[setting_count]['AUC'] = round(np.mean(aucs),4)
        evaluation_results[setting_count]['AUC_std'] = round(np.std(aucs),2)
        evaluation_results[setting_count]['AUPR'] = round(np.mean(auprs),2)
        evaluation_results[setting_count]['AUPR_std'] = round(np.std(auprs),2)
        setting_count+=1

print 'Experiment is done!\n'

# Save results:
f_out = open(work_path+'//results_'+name+'.txt','w')

f_out.write('threshold'+'\t'+'m'+'\t'+'AUC'+'\t'+'AUC std'+'\t'+'AUPR'+'\t'+'AUPR std\n')
for item in range(len(evaluation_results)):
    f_out.write(evaluation_results[item]['threshold']+'\t'+str(evaluation_results[item]['m'])+'\t'+str(evaluation_results[item]['AUC'])+'\t'+str(evaluation_results[item]['AUC_std'])+'\t'+str(evaluation_results[item]['AUPR'])+'\t'+str(evaluation_results[item]['AUPR_std'])+'\n')

f_out.close()

# Determine best parameters:
best_auc = np.max([evaluation_results[item]['AUC'] for item in evaluation_results])
index_best = [item for item in evaluation_results if evaluation_results[item]['AUC']==best_auc][0]
best_threshold = evaluation_results[index_best]['threshold']
best_m = evaluation_results[index_best]['m']
#%%
# Evaluate best parameters:
G,ligands,targets,layers = load_base_networks()
      
n = len(targets) # number of targets
m = len(ligands) # number of ligands

total_num = n*m # total number of all possible links
dt_links_all = np.zeros((total_num,2),np.int)

f_out = open(work_path+'//statistics_test_'+name+'.txt','w')

print 'Some statistics:'
print 'Nodes: ',G.number_of_nodes()
f_out.write('Nodes: '+str(G.number_of_nodes())+'\n')
print 'Edges: ',G.number_of_edges()
f_out.write('Edges: '+str(G.number_of_edges())+'\n')
print 'Connected components: ',nx.number_connected_components(G)
f_out.write('Connected components: '+str(nx.number_connected_components(G))+'\n')
print 'Ligands: ',m
f_out.write('Ligands: '+str(m)+'\n')
print 'Targets:',n
f_out.write('Targets: '+str(n)+'\n')
print 'Control sum: ',n+m,'\n'
f_out.write('Control sum: '+str(n+m)+'\n')

f_out.close()

# list of all possible dt-links (existing and none-existing):
count = 0
for t in range(n):
    for l in range(m):
        dt_links_all[count,0] = l
        dt_links_all[count,1] = t
        count+=1

# Load indexes of a test fold:
fold_indexes = np.load('data//internal_cv//'+name+'//test_nodes_'+str(fold)+'.npy')

interacting_links = []

# Inside a test fold remove all existing (interacting) links:
removed_links = [] # list of removed links
fold_links = [] # list of links in the fold

# Iterate over links in a fold:
for i in fold_indexes:
    dt_drug = 'l'+str(i[0])
    dt_target = 't'+str(i[1])
    fold_links.append((dt_drug,dt_target))
    
    # Check if link exists (one dt-layer is supported at the moment only):
    if G.has_edge(dt_drug,dt_target):
        
        # memorise link and its class:
        removed_links.append((dt_drug,dt_target,G.get_edge_data(dt_drug,dt_target)[dt_network]['weight']))
        
        if G.get_edge_data(dt_drug,dt_target)[dt_network]['weight'] == 1:
            interacting_links.append((dt_drug,dt_target))
        
        # remove it from G:
        G.remove_edge(dt_drug,dt_target)
        
f_out = open(work_path+'//results_test_'+name+'.txt','w')
        
print 'Best parameters evaluation\n'
f_out.write('Best parameters evaluation\n\n')
print 'Best threshold method: ',best_threshold
f_out.write('Best threshold method: '+best_threshold+'\n')
print 'Best m: ',str(best_m),'\n'
f_out.write('Best m: '+str(best_m)+'\n\n')

print 'Best AUC: ',str(evaluation_results[index_best]['AUC'])
f_out.write('Best AUC: '+str(evaluation_results[index_best]['AUC'])+'\n')
print 'Best AUC std: ',str(evaluation_results[index_best]['AUC_std'])
f_out.write('Best AUC std: '+str(evaluation_results[index_best]['AUC_std'])+'\n')
print 'Best AUPR: ',str(evaluation_results[index_best]['AUPR'])
f_out.write('Best AUPR: '+str(evaluation_results[index_best]['AUPR'])+'\n')
print 'Best AUPR std: ',str(evaluation_results[index_best]['AUPR_std']),'\n'
f_out.write('Best AUPR std: '+str(evaluation_results[index_best]['AUPR_std'])+'\n\n')

# Run partitioning:
groups = partitioning(G,best_m,best_threshold)

# Some statistics:
groups_unique,groups_sizes = np.unique(groups,return_counts=True)

print 'Maximum number of communities: ',2**best_m
f_out.write('Maximum number of communities: '+str(2**best_m)+'\n')
print 'Number of discovered communities: ',len(groups_unique)
f_out.write('Number of discovered communities: '+str(len(groups_unique))+'\n')
print 'Communities: ',groups_unique
f_out.write('Communities: '+str(groups_unique)+'\n')
print 'Communities sizes: ',groups_sizes,'\n'
f_out.write('Communities sizes: '+str(groups_sizes)+'\n\n')
       
# Dictionary of communities:
communities = {}

for group in groups_unique:
    communities[group] = []

for i in range(len(G.nodes())):
    communities[groups[i]].append(G.nodes()[i])
       
# Separate communities by type (ligand, target or both):
communities_lig = [group for group in communities.keys() if any(['l' in node for node in communities[group]])]
communities_tar = [group for group in communities.keys() if any(['t' in node for node in communities[group]])]

# Dictionary of link probabilities:
links_probs = {}

# Obtain real classes:
Y_test = [(couple in interacting_links) for couple in fold_links]

# Obtain link probabilities:
if matching_setting == 'com-com':
    
    # Match communities each with each and compute probabilities of none existing links in each combination:
    for comm_l in communities_lig:
        for comm_t in communities_tar:
            # Number of links between them:
            comm_links_exist = 0
            for node_l in communities[comm_l]:
                for node_t in communities[comm_t]:
                    if ('l' in node_l) and ('t' in node_t) and G.has_edge(node_l,node_t):
                        comm_links_exist+=1
            # Maximum possible number of links:
            comm_links_max = np.sum(['l' in el for el in communities[comm_l]])*np.sum(['t' in el for el in communities[comm_t]])
            # Link probability and vector of communities probabilities update:
            link_prob = comm_links_exist/float(comm_links_max)
            for node_l in communities[comm_l]:
                for node_t in communities[comm_t]:
                    if ('l' in node_l) and ('t' in node_t):
                        links_probs[node_l,node_t] = link_prob
    
    # Link probabilities:                    
    Y_scores = [links_probs[couple] for couple in fold_links]

elif 'node-com' in matching_setting:
    
    # Match ligands with target communities and compute probabilities of none existing links in each combination:
    for ligand in ligands:
        for comm_t in communities_tar:
            # Number of links between them:
            comm_links_exist = 0
            for node_t in communities[comm_t]:
                if ('t' in node_t) and G.has_edge(ligand,node_t):
                    comm_links_exist+=1
            # Maximum possible number of links:
            comm_links_max = np.sum(['t' in el for el in communities[comm_t]])
            # Link probability and vector of communities probabilities update:
            link_prob = comm_links_exist/float(comm_links_max)
            for node_t in communities[comm_t]:
                if ('t' in node_t):
                    links_probs[ligand,node_t] = [link_prob]
    
    # Match targets with ligand communities and compute probabilities of none existing links in each combination:
    for target in targets:
        for comm_l in communities_lig:
            # Number of links between them:
            comm_links_exist = 0
            for node_l in communities[comm_l]:
                if ('l' in node_l) and G.has_edge(node_l,target):
                    comm_links_exist+=1
            # Maximum possible number of links:
            comm_links_max = np.sum(['l' in el for el in communities[comm_l]])
            # Link probability and vector of communities probabilities update:
            link_prob = comm_links_exist/float(comm_links_max)
            for node_l in communities[comm_l]:
                if ('l' in node_l):
                    links_probs[node_l,target].append(link_prob)
    
    # Take an average between two local models:
    Y_scores = [np.mean(links_probs[couple]) for couple in fold_links]

# Evaluation by ROC-curve:
fpr,tpr,_ = metrics.roc_curve(Y_test,Y_scores)
auc_test = np.round(metrics.auc(fpr,tpr),4) # AUC score

# AUPR-curve:
precision,recall,_ = metrics.precision_recall_curve(Y_test,Y_scores) # PR-curve
aupr_test = np.round(metrics.auc(recall,precision),2) # AUPR score

print 'Test AUC score: ',auc_test
f_out.write('Test AUC score: '+str(auc_test)+'\n')
print 'Test AUPR score: ',str(aupr_test),'\n'
f_out.write('Test AUPR score: '+str(aupr_test)+'\n\n')

# save results in .xls format
f_out.write('Line to copy in .xls format:\n')
f_out.write(fold+'\t'+best_threshold+'\t'+str(best_m)+'\t'+str(evaluation_results[index_best]['AUC'])+'\t'+str(evaluation_results[index_best]['AUC_std'])+'\t'+str(evaluation_results[index_best]['AUPR'])+'\t'+str(evaluation_results[index_best]['AUPR_std'])+'\t'+str(auc_test)+'\t'+str(aupr_test)+'\n')

f_out.close()

print 'Evaluation is finished!'
#%%
