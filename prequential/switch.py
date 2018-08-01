import numpy as np
import pickle as pkl

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse

import pdb
        
def plot_metrics_samples(modelsscoreslist, namefile, subplots=None,
                         num_classes=10, datasetshape=50000):
    maxindexes = max(m["indexes"][-1] for m in modelsscoreslist)
    
        

    if subplots is None:
        subplots = ["val_loss", "loss", "val_acc", "acc", "cost", "comprate", "costlab"]
    n_subplots = len(subplots)
    #gs = gridspec.GridSpec(3, 1)
    #gs.update(left=0.05, right=0.48, wspace=0.05)
    #fig, axes = plt.subplots(n_subplots, 1, figsize=(8,3*n_subplots))
    fig, axes = plt.subplots(n_subplots // 2, 2, figsize=(10, 5))


    # Loss plot
    def loss_subplot(losskey, title, ax):
        #ax.set_title(title)
        for m in [m for m in modelsscoreslist if losskey in m]:
            mloss = m[losskey]
            ax.plot(m["indexes"], mloss, label=m["shortdescription"],
                      linewidth=1., alpha=0.7, color=m.get("color"),
                      linestyle=m.get("linestyle"))        
        ax.set_yscale('log')
        ax.set_xlim([0., datasetshape])
        #ax.set_xlabel('Mini-batch-number')
        ax.set_ylabel('Loss (log-scale)')
        ax.set_xlabel('Number of samples')
        ax.legend(loc="upper right",fontsize=8)
        #ax.set_xscale('log')
        ax.get_yaxis().set_label_coords(-0.1,0.5)

        
    
    def acc_subplot(acckey, title, ax):
        #ax.set_title(title)
        #ax_acc.grid(axis='y', color='k', linewidth=0.2)
        for m in [m for m in modelsscoreslist if acckey in m]:
            ax.plot(m["indexes"], m[acckey], 
                label=m["shortdescription"], linewidth=1., alpha=0.7,
                color=m.get("color"), linestyle=m.get("linestyle"))
        ax.set_xlim([0., datasetshape])
        ax.set_ylim([0., 1.])
        #ax_acc.set_xlabel('Mini-batch-number')
        ax.set_ylabel('Accuracy on the next\ndata pack (%)')
        ax.set_xlabel('Number of samples')
        ax.legend(loc="lower right",fontsize=8., ncol=2)
        ax.get_yaxis().set_label_coords(-0.1,0.5)

    
    # Cost plot
    def cost_subplot(title, ax):
        #ax.set_title(title)
        
        ##### A SUPPRIMER
        _, costbase, _, _ = modelsscoreslist[0]["cost"]
        for m in [m for m in modelsscoreslist if "cost" in m]:
            indexes_cost, cost, _, _ = m["cost"]           
            ax.plot(indexes_cost, (cost - costbase)/1000, 
                        label=m["shortdescription"], 
                        linewidth=1., alpha=0.7,
                        color=m.get("color"), 
                        linestyle=m.get("linestyle"))
            print(m["shortdescription"], cost[-1])
            #ax.text(indexes_cost[-1] + 100, cost[-1], 
            #        str(int(cost[-1])), fontsize=6., 
            #        #color=m.get("color"),
            #        )
            
            #ax.legend(loc="lower left",fontsize=8.)
        
        
        #ax_loss.set_xlabel('Mini-batch-number')
        ax.set_ylabel('Cumulative encoding cost\n(difference with uniform) (kbits)')
        #ax.set_xlabel('Number of samples')
        ax.set_xlim([0., datasetshape])
        ax.get_yaxis().set_label_coords(-0.1,0.5)

        
    def costlab_subplot(title, ax):
        #ax.set_title(title)
        for m in [m for m in modelsscoreslist if "cost" in m]:
            indexes_cost, _, _, costlab = m["cost"]           
            ax.plot(indexes_cost, costlab, 
                        label=m["shortdescription"], 
                        linewidth=1., alpha=0.7,
                        color=m.get("color"), 
                        linestyle=m.get("linestyle"))
        #ax.legend(loc="upper right",fontsize=8.)            
    
        #ax_loss.set_xlabel('Mini-batch-number')
        ax.set_ylabel('Encoding cost per \nsample (bits)')
        ax.set_ylim([0., 2*np.log2(10)])
        #ax.set_xlabel('Number of samples')
        ax.set_xlim([0., datasetshape])
        #ax.set_yscale('log')
        ax.get_yaxis().set_label_coords(-0.1,0.5)


    def compressionrate_subplot(title, ax):
        #ax.set_title(title)
        for m in [m for m in modelsscoreslist if "cost" in m]:
            indexes_cost, _, comprate, _ = m["cost"]           
            ax.plot(indexes_cost, comprate, 
                        label=m["shortdescription"], 
                        linewidth=1., alpha=0.7,
                        color=m.get("color"), 
                        linestyle=m.get("linestyle"))
    
        ax.set_xlabel('Number of samples')
        ax.set_ylabel('Compression ratio')
        ax.set_xlim([0., datasetshape])
        ax.set_ylim([0., 2.])
        #ax.set_yscale('log')
        #ax.legend(bbox_to_anchor=(0., -.6, 1., -1.6), ncol=2, mode="expand", loc=3, borderaxespad=0., fontsize=10.)
        ax.get_yaxis().set_label_coords(-0.1,0.5)

    
 
    
    for subp, ax in zip(subplots, axes.flat):
        if subp == "val_loss":
            loss_subplot("val_loss", "Loss (evaluated on the next pack of data)", ax)
        if subp == "loss":
            loss_subplot("loss", "Loss (train)", ax)
        if subp == "val_acc":
            acc_subplot("val_acc", "Accuracy (evaluated on the next pack of data)", ax)
        if subp == "acc":   
            acc_subplot("acc", "Accuracy (train)", ax)
        if subp == "cost":
            cost_subplot("Cumulative encoding cost (difference with uniform encoding cost)", ax)
        if subp == "comprate":
            compressionrate_subplot("Compression rate", ax)
        if subp == "costlab":
            costlab_subplot("Encoding cost for each label", ax)
    
    
    
    
    fig.tight_layout()
    plt.savefig(namefile, format="eps")
    
    
def costfun(indexes, loss, initial_cost, datasetshape, interpolation=False):
    
    indexes_cost = np.arange(datasetshape + 1)
    uniform_cost = initial_cost * indexes_cost
    loss_cost = np.zeros(datasetshape + 1)
    for k, idx in enumerate(indexes):
        if k == len(indexes) - 1:
            maxidx = datasetshape + 1 
            loss_cost[idx:] = loss[-1]
        else:
            for t in range(idx, indexes[k+1]):
                if interpolation:
                    loss_cost[t] = loss[k] + (t - idx) / (indexes[k+1] - idx) * \
                        (loss[k+1] - loss[k])
                else:
                    loss_cost[t] = loss[k]    
    #loss_cost = np.zeros(len(loss) +1 )
    loss_cost[:indexes[0]] = initial_cost
    #loss_cost[1:] = loss
    
    cost = loss_cost - initial_cost
    
    compressionbound = loss_cost.cumsum() / uniform_cost
    #cost[1:] = (loss_cost - initial_cost) * (indexes_cost[1:] - indexes_cost[:-1])
    cost = cost.cumsum()
    
    #cost = cost -
    return indexes_cost, cost, compressionbound, loss_cost


def switch_loss(modelsscoreslist, num_classes, datasetshape, interpolate=False):
    initial_cost = np.log(num_classes) / np.log(2) # HERE IN BITS
    maxindexes = datasetshape
        
    indexes = list(range(maxindexes))
    switchloss = [initial_cost for _ in range(maxindexes)] 
    for m in modelsscoreslist:
        mloss = m["val_loss"]
                
        for k, idx in enumerate(m["indexes"]):
            if k == len(m["indexes"]) - 1:
                maxrange = maxindexes
            else:
                maxrange = m["indexes"][k+1]
            for t in range(idx, maxrange):
                if k == len(m["indexes"]) - 1:
                    switchloss[t] = min(mloss[k], switchloss[t])
                else:
                    if interpolate:
                        interp = m["val_loss"][k] + \
                            (t - idx) / (m["indexes"][k+1] - idx) * \
                            (m["val_loss"][k+1] - m["val_loss"][k])
                    else:
                        interp = mloss[k]
                    switchloss[t] = min(interp, switchloss[t])
            
    switchdict = {"description":"Switch", "shortdescription":"Switch",
        "indexes":indexes,
        "val_loss":switchloss,
        "color":"r",
        "linestyle":"--", 
        #"acc_test":(1/num_classes)*np.ones(maxindexes),
        #"loss_train":np.log(num_classes)*np.ones(maxindexes),
        #"acc_train":(1/num_classes)*np.ones(maxindexes),
        }
        
    return switchdict
    
    
    
    

        
        

def makemodelscoreslist(modellist, num_classes=10, datasetshape=50000, autoswitch="none"):
    modelsscoreslist = []
    maxindexes = max(m["indexes"][-1] for m in modellist)
    
    uniform = {"description":"Uniform random", "shortdescription":"uniform",
        "indexes":np.arange(maxindexes),
        "val_loss":np.log2(num_classes)*np.ones(maxindexes), 
        "val_acc":(1/num_classes)*np.ones(maxindexes),
        "loss":np.log2(num_classes)*np.ones(maxindexes),
        "acc":(1/num_classes)*np.ones(maxindexes),
        "linestyle":":",
        "color":"k"}
    
    modelsscoreslist.append(uniform)
    
    if True: #if in bits
        for m in modellist:
            if "histories" in m:
                for key in ["loss", "val_loss"]:
                    for h in m["histories"]:
                        h[key] /= np.log(2)
            else:
                if "loss_train" in m:
                    m["loss_train"] = m["loss_train"] / np.log(2)
               
                m["loss_test"] = m["loss_test"] / np.log(2)

                
        
        
        
    for m in modellist:
        newm = {}
        for key in ["indexes", "description", "shortdescription"]:
            newm[key] = m[key]
        
        
        if "histories" in m:
            for key in ["loss", "acc", "val_loss", "val_acc"]:
                newm[key] = [h[key][-1] for h in m["histories"]]
            if autoswitch == "none" or autoswitch == "both":
                modelsscoreslist.append(newm)
        else:
            if "loss_train" in m:
                newm["loss"] = m["loss_train"] 
            if "acc_train" in m:
                newm["acc"] = m["acc_train"]
            newm["val_loss"] = m["loss_test"] 
            newm["val_acc"] = m["acc_test"]
            modelsscoreslist.append(newm)
        
            
    countselfsw = 0
    for m in [m for m in modellist if "histories" in m]:
        autoswitchm = {}
        autoswitchm["indexes"] = m["indexes"]
        autoswitchm["description"] = m["description"] + " +autoswitch"
        autoswitchm["shortdescription"] = m["shortdescription"] + "+SelfSw"
        autoswitchm["linestyle"] = "--"
        
        autoswitchm['color'] = 'C'+str(countselfsw)
        
        for key in ["loss", "acc", "val_loss", "val_acc"]:
            autoswitchm[key] = []
        
        tmp=0
        for h in m["histories"]:
            
            
            bestl = np.inf
            bestk = 0
            for (k, l) in enumerate(h["val_loss"]):
                if l < bestl:
                    bestl = l
                    bestk = k
            print(m["shortdescription"], m["indexes"][tmp], bestk)        
            for key in ["loss", "acc", "val_loss", "val_acc"]:
                autoswitchm[key].append(h[key][bestk])
            tmp += 1
        if autoswitch == "as" or autoswitch == "both":      
            modelsscoreslist.append(autoswitchm)

        countselfsw += 1
        
            
        
    switchscores = switch_loss(modelsscoreslist, num_classes, datasetshape)
    
    if autoswitch == "both" or autoswitch == "sw":
        pass
        #modelsscoreslist.append(switchscores)
    
    for m in modelsscoreslist:
        m["cost"] = costfun(m["indexes"], m["val_loss"], 
                            np.log(num_classes)/np.log(2), datasetshape)
                            
    return modelsscoreslist                        
    
  

with open("metrics.pkl", "rb") as f:
    modelsscoreslist = pkl.load(f)


for m in modelsscoreslist:
    print(m["description"])
    
    

newmodelsscoreslist = makemodelscoreslist(modelsscoreslist, autoswitch="both", datasetshape=50000)

for m in newmodelsscoreslist:
    print(m["shortdescription"])


subplots = ["costlab",  "cost", "val_acc",   "comprate", ]
plot_metrics_samples(newmodelsscoreslist, "cifarscores.eps", subplots, num_classes=10, datasetshape=50000)
  
