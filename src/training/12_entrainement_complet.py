#!/usr/bin/env python3
# =====================================================================
# 12_entrainement_complet.py
# ENTRAINEMENT COMPLET (grande cohorte, float16/memmap).
# Ajoute par rapport au script precedent :
#   - metrique JACCARD (diagnostic)
#   - sauvegarde des PREDICTIONS  -> pour la fairness (script 20)
#   - sauvegarde des cartes d ATTENTION -> pour l IoU (script 21)
#   - lit les patches en float16 memmap (grande cohorte)
# =====================================================================

import os, json, random, argparse
import numpy as np, pandas as pd, torch
import torch.nn as nn
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, precision_score, recall_score, jaccard_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.expanduser('~/scratch/MultimodalVLM')
PROC = os.path.join(BASE, 'data/processed')
RAW  = os.path.join(BASE, 'data/raw')
RES  = os.path.join(BASE, 'resultats')
os.makedirs(RES, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--maladie', default='Pleural Effusion')
parser.add_argument('--ehr', default='enrichi', choices=['simple','enrichi'])
parser.add_argument('--lambda_c', type=float, default=2.0)
parser.add_argument('--epochs', type=int, default=40)
args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('=== ENTRAINEMENT COMPLET ===', flush=True)
print('Maladie:', args.maladie, '| EHR:', args.ehr, '| lambda:', args.lambda_c, '| device:', device, flush=True)

# ---------- patches float16 memmap ----------
patch_f16 = os.path.join(PROC, 'image_patches_f16.npy')
patch_f32 = os.path.join(PROC, 'image_patches_clean.npy')
if os.path.exists(patch_f16):
    patches = np.load(patch_f16, mmap_mode='r')   # memmap, ne charge pas tout en RAM
    print('Patches float16 memmap :', patches.shape, flush=True)
else:
    patches = np.load(patch_f32, mmap_mode='r')
    print('Patches float32 memmap :', patches.shape, flush=True)

index = pd.read_csv(os.path.join(PROC, 'image_index_clean.csv'))
row_of = {d:i for i,d in enumerate(index['dicom_id'])}

if args.ehr=='enrichi' and os.path.exists(os.path.join(RAW,'cohort_ehr_text_enriched.csv')):
    ehr = pd.read_csv(os.path.join(RAW,'cohort_ehr_text_enriched.csv'))
else:
    ehr = pd.read_csv(os.path.join(PROC,'cohort_ehr_text_clean.csv'))

chex = pd.read_csv(os.path.join(RAW,'labels_detection.csv'))
chex['label']=(chex[args.maladie]==1.0).astype(int)
labels=dict(zip(chex['study_id'],chex['label']))
index['label']=index['study_id'].map(labels)
index=index.dropna(subset=['label']).reset_index(drop=True); index['label']=index['label'].astype(int)

random.seed(42); pats=list(index['subject_id'].unique()); random.shuffle(pats)
n=len(pats); a=int(0.70*n); b=int(0.85*n)
p_tr,p_va,p_te=set(pats[:a]),set(pats[a:b]),set(pats[b:])
def mk(ps):
    s=index[index.subject_id.isin(ps)]
    return [(row_of[r['dicom_id']],r['subject_id'],r['dicom_id'],int(r['label']))
            for _,r in s.iterrows() if r['dicom_id'] in row_of]
train_s,val_s,test_s=mk(p_tr),mk(p_va),mk(p_te)
print(f'Patients {len(pats)} | train {len(train_s)} / val {len(val_s)} / test {len(test_s)}', flush=True)

# ---------- texte : Bio_ClinicalBERT ----------
from transformers import AutoTokenizer, AutoModel
TXT='emilyalsentzer/Bio_ClinicalBERT'
tok=AutoTokenizer.from_pretrained(TXT); bert=AutoModel.from_pretrained(TXT).eval().to(device)
DIM=bert.config.hidden_size
text_emb={}
for subj in index['subject_id'].unique():
    sub=ehr[ehr.subject_id==subj]
    txt=sub['texte'].iloc[0] if len(sub) else 'patient .'
    inp=tok(txt,return_tensors='pt',truncation=True,max_length=256).to(device)
    with torch.no_grad(): out=bert(**inp)
    text_emb[subj]=out.last_hidden_state[0].cpu()
del bert; torch.cuda.empty_cache()
print('Texte encode.', flush=True)

class Modele(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.proj=nn.Linear(768,dim)
        self.cross=nn.MultiheadAttention(dim,8,batch_first=True)
        self.h_img=nn.Linear(dim,1); self.h_txt=nn.Linear(dim,1); self.h_fus=nn.Linear(dim,1)
    def forward(self,p,t,retour_attn=False):
        img=self.proj(p)
        fus,attn=self.cross(t,img,img,need_weights=retour_attn,average_attn_weights=True)
        out=(self.h_fus(fus.mean(1)).squeeze(-1),self.h_img(img.mean(1)).squeeze(-1),self.h_txt(t.mean(1)).squeeze(-1))
        return (out, attn) if retour_attn else out

def evaluer(yt,yp):
    yp2=[1 if x>0.5 else 0 for x in yp]
    return {'AUROC':roc_auc_score(yt,yp),'AUPRC':average_precision_score(yt,yp),
            'F1':f1_score(yt,yp2,zero_division=0),'Precision':precision_score(yt,yp2,zero_division=0),
            'Recall':recall_score(yt,yp2,zero_division=0),
            'Jaccard':jaccard_score(yt,yp2,zero_division=0)}   # <-- Jaccard ajoute

def get(i):
    return torch.tensor(np.asarray(patches[i]),dtype=torch.float32)

def entrainer(lam, sauver_attn=False):
    torch.manual_seed(0); model=Modele(DIM).to(device)
    npos=sum(y for *_,y in train_s); nneg=len(train_s)-npos
    pw=torch.tensor([max(nneg,1)/max(npos,1)]).to(device)
    bce=nn.BCEWithLogitsLoss(pos_weight=pw); mse=nn.MSELoss()
    opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    best=1e9; bs=None; wait=0
    for ep in range(args.epochs):
        random.shuffle(train_s); model.train()
        for (i,subj,dic,y) in train_s:
            pi=get(i).unsqueeze(0).to(device); te=text_emb[subj].unsqueeze(0).to(device)
            yt=torch.tensor([float(y)]).to(device); fus,img,txt=model(pi,te)
            loss=bce(fus,yt)+bce(img,yt)+bce(txt,yt)
            if lam>0: loss=loss+lam*mse(torch.sigmoid(img),torch.sigmoid(txt))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); lv=0
        with torch.no_grad():
            for (i,subj,dic,y) in val_s:
                pi=get(i).unsqueeze(0).to(device); te=text_emb[subj].unsqueeze(0).to(device)
                lv+=bce(model(pi,te)[0],torch.tensor([float(y)]).to(device)).item()
        lv/=max(len(val_s),1)
        if lv<best: best=lv; bs={k:v.clone() for k,v in model.state_dict().items()}; wait=0
        else:
            wait+=1
            if wait>=6: break
    model.load_state_dict(bs); model.eval()
    yt,yp,subj_list,dic_list,attn_list=[],[],[],[],[]
    with torch.no_grad():
        for (i,subj,dic,y) in test_s:
            pi=get(i).unsqueeze(0).to(device); te=text_emb[subj].unsqueeze(0).to(device)
            if sauver_attn:
                (fus,img,txt),attn=model(pi,te,retour_attn=True)
                # attention moyenne du texte vers les 196 patches image
                attn_list.append(attn.mean(1).squeeze(0).cpu().numpy())
            else:
                fus,img,txt=model(pi,te)
            yt.append(y); yp.append(torch.sigmoid(fus).item())
            subj_list.append(subj); dic_list.append(dic)
    m=evaluer(yt,yp)
    return m,(yt,yp,subj_list,dic_list,attn_list)

tag=f'{args.maladie.replace(" ","_")}_{args.ehr}'

# baseline + L_consist
print('\n--- Baseline (lambda=0) ---', flush=True)
m0,_=entrainer(0.0)
print('  ',{k:round(v,3) for k,v in m0.items()}, flush=True)
print('--- +L_consist ---', flush=True)
m1,(yt,yp,subj_list,dic_list,attn_list)=entrainer(args.lambda_c, sauver_attn=True)
print('  ',{k:round(v,3) for k,v in m1.items()}, flush=True)

# tableau resultats (avec Jaccard)
pd.DataFrame({'Baseline':m0,f'+L_consist(λ={args.lambda_c})':m1}).T.round(3)\
  .to_csv(os.path.join(RES,f'complet_{tag}.csv'))

# predictions -> pour fairness (script 20)
pd.DataFrame({'subject_id':subj_list,'y_true':yt,'y_pred':yp})\
  .to_csv(os.path.join(RES,f'predictions_{tag}.csv'),index=False)

# attention -> pour IoU (script 21)
if attn_list:
    np.save(os.path.join(RES,'attention_maps.npy'), np.array(attn_list))
    pd.DataFrame({'dicom_id':dic_list}).to_csv(os.path.join(RES,'attention_index.csv'),index=False)

print('\n=== TERMINE ===', flush=True)
print('Fichiers : complet_'+tag+'.csv (avec Jaccard), predictions_'+tag+'.csv, attention_maps.npy', flush=True)
print('-> lancer ensuite 20_evaluation_fairness.py et 21_explicabilite.py', flush=True)
