from typing import Optional
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from ttbar.network import QuarkTripletNetwork
from ttbar.options import Options
from ttbar.dataset import TTBarDataset
from tqdm import tqdm
import numpy as np
import h5py
from skhep.math.vectors import LorentzVector
import matplotlib.pyplot as plt
import math

def to3d( idx, maxjets ):
    z = int( idx / (maxjets * maxjets) )
    idx -= ( z*maxjets*maxjets )
    y = int( idx / maxjets )
    x = int( idx % maxjets )
    return (z,y,x)

vto3d = np.vectorize(to3d,excluded=['maxjets'])

def evaluate(input_file,test_file,mask):
    hparams = Options(input_file)
    model = QuarkTripletNetwork(hparams)
    checkpoint_path = './lightning_logs/version_2/checkpoints/'
    #checkpoint = 'epoch=49-step=949.ckpt'
    checkpoint = 'epoch=149-step=8099.ckpt'
    checkpoint_after_training = torch.load(checkpoint_path+checkpoint)
    model.load_state_dict(checkpoint_after_training['state_dict'])
    device = torch.device('cuda:0')
    model.to(device)    
    model.eval()

    batch_size = model.options.batch_size
    num_workers = model.options.num_dataloader_workers
    max_jets = model.training_dataset.max_jets
    mean = model.training_dataset.mean
    std = model.training_dataset.std
    #test_file = "./data/tr_files/tteemm_sig_new_test.h5"
    #test_file = "./data/tteemm_allhadronic_train.h5"
    test_dataset = TTBarDataset(hdf5_path=test_file,event_mask=mask,max_jets=max_jets)
    test_dataset.normalize(mean,std)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 shuffle=False
    )

    #test_dataloader_train = model.testing_dataloader()
    
    left_map = []
    right_map = []

    left_preds_indices = []
    right_preds_indices = []

    left_targs_indices = []
    right_targs_indices = []

    tk0 = tqdm(test_dataloader, desc="Iteration")

    for step, batch in enumerate(tk0):
        x, targets, mask = batch
        #targets[0].to(device)
        #targets[1].to(device)
        
        #sw_targets = model.swap_antiparticle_targets(targets)

        ### for loading to gpu somehow
        x = x.to(device, dtype=torch.float)
        mask = mask.to(device, dtype=torch.bool)

        predictions = model(x,mask)

        left_predictions = predictions[0].clone().cpu()
        right_predictions = predictions[1].clone().cpu()
        
        left_targets = targets[0].clone()
        right_targets = targets[1].clone()
        #sw_left_targets = sw_targets[0].clone()
        #sw_right_targets = sw_targets[1].clone()
        
        batch_size, max_jets, _, _ = left_predictions.shape
        
        # Zero out the lower triangle to make accuracy calculation easier
        # Both targets and predictions should be symmetric anyway
        for i in range(max_jets):
            for j in range(i):
                left_predictions[:, i, j, :] = 0
                right_predictions[:, i, j, :] = 0
                left_targets[:, i, j, :] = 0
                right_targets[:, i, j, :] = 0
        
        left_targets = left_targets.view(batch_size, -1).argmax(1)
        right_targets = right_targets.view(batch_size, -1).argmax(1)
        
        #sw_left_targets = sw_left_targets.view(batch_size, -1).argmax(1)
        #sw_right_targets = sw_right_targets.view(batch_size, -1).argmax(1)
        
        left_predictions = left_predictions.view(batch_size, -1).argmax(1)
        right_predictions = right_predictions.view(batch_size, -1).argmax(1)
        
        #left_correct = (left_predictions==left_targets) | (left_predictions==sw_left_targets)
        #right_correct = (right_predictions==right_targets) | (right_predictions==sw_right_targets)

        #left_map.extend(left_correct.tolist())
        #right_map.extend(right_correct.tolist())

        left_preds_indices.extend(left_predictions.tolist())
        right_preds_indices.extend(right_predictions.tolist())

        left_targs_indices.extend(left_targets.tolist())
        right_targs_indices.extend(right_targets.tolist())

        #if left_predictions == left_targets | left_predictions == sw_left_targets:
        #    left_correct = True
        #if right_predictions == right_targets | right_predictions == sw_right_targets:
        #    right_correct = True
        #
        #if left_correct | right_correct:
        #    either += 1
        #if left_correct and right_correct:
        #    both += 1
        #break

    left_map = np.asarray(left_map)
    right_map = np.asarray(right_map)

    left_preds_indices = np.array(vto3d(np.asarray(left_preds_indices),maxjets=max_jets)).T
    right_preds_indices = np.array(vto3d(np.asarray(right_preds_indices),maxjets=max_jets)).T
    left_targs_indices = np.array(vto3d(np.asarray(left_targs_indices),maxjets=max_jets)).T
    right_targs_indices = np.array(vto3d(np.asarray(right_targs_indices),maxjets=max_jets)).T
    #print(left_preds_indices[20],
    #      right_preds_indices[20],
    #      left_targs_indices[20],
    #      right_targs_indices[20],
    #      left_map[20],
    #      right_map[20]
    #)

    #either = left_map | right_map
    #both = left_map & right_map
    #print(len(either))
    #print("Either: ", (either==True).sum() /float(len(either)) )
    #print("Both: ", (both==True).sum()/float(len(either)))
    return (left_preds_indices,right_preds_indices,left_targs_indices,right_targs_indices)

def read_file(test_file,mask):

    events = []

    with h5py.File(test_file,"r") as f:
        keys = f.keys()
        jet_pt = f.get("jet_pt")[:]
        jet_eta = f.get("jet_eta")[:]
        jet_phi = f.get("jet_phi")[:]
        jet_m = f.get("jet_mass")[:]
        parton_pt = f.get("parton_pt")[:]
        parton_eta = f.get("parton_eta")[:]
        parton_phi = f.get("parton_phi")[:]
        parton_m = f.get("parton_mass")[:]
        jet_parton_index = f.get("jet_parton_index")[:]
        parton_jet_index = f.get("parton_jet_index")[:]
        N_match_top_in_event = f.get("N_match_top_in_event")[:]
        for i, jpts in enumerate(jet_pt):
            event = {
                'Ntops': -1,
                'jets': [],
                'partons' : [],
                'parton_jet_indices': [],
                'jet_parton_indices': []
            }
            if mask != None:
                if int(N_match_top_in_event[i]) != mask:
                    continue
            jets = []
            partons = []
            for j, pt in enumerate(jpts):
                temp_jet = LorentzVector()
                temp_jet.setptetaphim(jet_pt[i][j],jet_eta[i][j],jet_phi[i][j],jet_m[i][j])
                jets.append(temp_jet)
            for p in range(0,6):
                temp_parton = LorentzVector()
                temp_parton.setptetaphim(parton_pt[i][p],parton_eta[i][p],parton_phi[i][p],parton_m[i][p])
                partons.append(temp_parton)
            event['Ntops'] = N_match_top_in_event[i]
            event['jets'] = jets
            event['partons'] = partons
            event['parton_jet_indices'] = parton_jet_index[i]
            event['jet_parton_indices'] = jet_parton_index[i]
            events.append(event)
    
    return events
        
#def plot_events(events):
    
def find_correct_indices(event):
    
    top1 = LorentzVector()
    top2 = LorentzVector()

    firstTripletEx = True
    secondTripletEx = True

    firstTriplet = (-1,-1,-1)
    secondTriplet = (-1,-1,-1)

    indx = event['parton_jet_indices']

    for i,pji in enumerate(indx):
        if pji == -1 or np.isnan(pji):
            if i < 3:
                firstTripletEx = False
            else:
                secondTripletEx = False

    l0 = int(event['SPA_left_targs'][2])
    l1 = int(event['SPA_left_targs'][0])
    l2 = int(event['SPA_left_targs'][1])

    r0 = int(event['SPA_right_targs'][2])
    r1 = int(event['SPA_right_targs'][0])
    r2 = int(event['SPA_right_targs'][1])

    if firstTripletEx:
        firstTriplet = ( int(indx[0]), int(indx[1]), int(indx[2]) )
        if int(indx[0]) != l0 and int(indx[0]) != r0:
            exit("Problem matching first triplet")
        if int(indx[0]) == l0:
            if int(indx[1]) != l1 and int(indx[1]) != l2:
                exit("Problem matching first triplet")
            if int(indx[2]) != l1 and int(indx[2]) != l2:
                exit("Problem matching first triplet")
        elif int(indx[0]) == r0:
            if int(indx[1]) != r1 and int(indx[1]) != r2:
                exit("Problem matching first triplet")
            if int(indx[2]) != r1 and int(indx[2]) != r2:
                exit("Problem matching first triplet")
    if secondTripletEx:
        secondTriplet = ( int(indx[3]), int(indx[4]), int(indx[5]) )
        if int(indx[3]) != int(event['SPA_left_targs'][2]) and int(indx[0]) != int(event['SPA_right_targs'][2]):
            exit("Problem matching second triplet")
        if int(indx[3]) == l0:
            if int(indx[4]) != l1 and int(indx[4]) != l2:
                exit("Problem matching first triplet")
            if int(indx[5]) != l1 and int(indx[5]) != l2:
                exit("Problem matching first triplet")
        elif int(indx[3]) == r0:
            if int(indx[4]) != r1 and int(indx[4]) != r2:
                exit("Problem matching first triplet")
            if int(indx[5]) != r1 and int(indx[5]) != r2:
                exit("Problem matching first triplet")

    event['firstTriplet'] = firstTriplet
    event['secondTriplet'] = secondTriplet

    #print( firstTriplet, secondTriplet, event['SPA_left_targs'], event['SPA_right_targs'] )
    #print(event['parton_jet_indices'],firstTriplet,secondTriplet,event['Ntops'])
    #print(firstTriplet[1])

def get_matched_reco_tops(event):
    
    firstTop = LorentzVector()
    secondTop = LorentzVector()
    jets = event['jets']
    if -1 not in event['firstTriplet']:
        firstTop = jets[ event['firstTriplet'][0] ]
        firstTop += jets[ event['firstTriplet'][1] ]
        firstTop += jets[ event['firstTriplet'][2] ]
    if -1 not in event['secondTriplet']:
        secondTop = jets[ event['secondTriplet'][0] ]
        secondTop += jets[ event['secondTriplet'][1] ]
        secondTop += jets[ event['secondTriplet'][2] ]
    
    #print(firstTop.pt,secondTop.pt,firstTop.m,secondTop.m)
    if firstTop.pt > secondTop.pt:
        event['leadTop'] = firstTop
        event['subleadTop'] = secondTop
    else:
        event['leadTop'] = secondTop
        event['subleadTop'] = firstTop

def get_pred_reco_tops(event):
    firstTop = LorentzVector()
    secondTop = LorentzVector()
    jets = event['jets']
    l0 = int(event['SPA_left_preds'][2])
    l1 = int(event['SPA_left_preds'][0])
    l2 = int(event['SPA_left_preds'][1])

    r0 = int(event['SPA_right_preds'][2])
    r1 = int(event['SPA_right_preds'][0])
    r2 = int(event['SPA_right_preds'][1])

    left_targs = event['firstTriplet']
    right_targs = event['secondTriplet']

    firstTop = jets[l0]+jets[l1]+jets[l2]
    secondTop = jets[r0]+jets[r1]+jets[r2]

    firstCorrect = False
    secondCorrect = False

    if (l0,l1,l2) == left_targs or (l0,l1,l2) == right_targs:
        firstCorrect = True
    elif (l0,l2,l1) == left_targs or (l0,l2,l1) == right_targs:
        firstCorrect = True
    if (r0,r1,r2) == left_targs or (r0,r1,r2) == right_targs:
        secondCorrect = True
    elif (r0,r2,r1) == left_targs or (r0,r2,r1) == right_targs:
        secondCorrect = True

    if firstTop.pt > secondTop.pt:
        event['leadPredTop'] = firstTop
        event['subLeadPredTop'] = secondTop
        event['leadPredCorr'] = firstCorrect
        event['subLeadPredCorr'] = secondCorrect
    else:
        event['leadPredTop'] = secondTop
        event['subLeadPredTop'] = firstTop
        event['leadPredCorr'] = secondCorrect
        event['subLeadPredCorr'] = firstCorrect
    

def attach_preds(events,left_preds,right_preds,left_targs,right_targs):
    if len(events) != len(left_preds):
        return None
    if len(events) != len(right_preds):
        return None
    if len(events) != len(left_targs):
        return None
    if len(events) != len(right_targs):
        return None

    for i,e in enumerate(events):
        e['SPA_left_preds'] = left_preds[i]
        e['SPA_right_preds'] = right_preds[i]
        e['SPA_left_targs'] = left_targs[i]
        e['SPA_right_targs'] = right_targs[i]
    

def plot(events):
    
    unmatchable = []
    incorrect = []
    correct = []

    for e in events:
        if e['Ntops'] == 0:
            unmatchable.append(e['leadPredTop'].pt)
        else:
            if e['leadPredCorr']:
                correct.append(e['leadPredTop'].pt)
            else:
                incorrect.append(e['leadPredTop'].pt)

    unmatchable = np.asarray(unmatchable)
    incorrect = np.asarray(incorrect)
    correct = np.asarray(correct)
    
    d_min = min( [unmatchable.min(),incorrect.min(),correct.min()] )
    d_max = max( [unmatchable.max(),incorrect.min(),correct.max()] )

    d_min = 50
    d_max = 500

    n_bins = 45

    fig = plt.figure()
    plt.hist( [unmatchable,incorrect,correct], bins=n_bins, stacked=True, range=(d_min,d_max), )
    fig.savefig( "lead_top_pt.pdf" )
    

if __name__ == '__main__':
    train_file = "./data/tr_files/tteemm_sig_new_train.h5"
    test_file = "./data/tr_files/tteemm_sig_new_test.h5"
    mask = None

    left_preds, right_preds, left_targs, right_targs = evaluate(train_file,test_file,mask)

    events = read_file(test_file,mask)
    attach_preds(events,left_preds,right_preds,left_targs,right_targs)
    for e in events:
        find_correct_indices(e)
        #get_matched_reco_tops(e)
        get_pred_reco_tops(e)
    plot(events)
    
