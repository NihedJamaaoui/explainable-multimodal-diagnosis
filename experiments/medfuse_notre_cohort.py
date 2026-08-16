#!/usr/bin/env python3
# =====================================================================
# medfuse_notre_cohort.py
# COMPARAISON NIVEAU 1 : la fusion LSTM de MedFuse sur NOTRE cohort exact.
#
# But : comparer l'ARCHITECTURE de fusion (MedFuse LSTM vs notre cross-attention)
#       sur EXACTEMENT nos donnees, notre split, nos 5 pathologies.
#
# Idee : on reprend le mecanisme central de MedFuse (models/fusion.py) :
#        empiler ehr_feats et cxr_feats comme une SEQUENCE de longueur 2,
#        la passer dans un LSTM, puis classifier. On lui donne NOS features
#        (patches BiomedCLIP moyennes + EHR Bio_ClinicalBERT moyenne),
#        les memes que voit notre modele.
#
# Sortie : AUROC / AUPRC macro sur notre test 15% -> ligne de comparaison directe.
# =====================================================================

import os, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import random

BASE=os.path.expanduser('~/scratch/MultimodalVLM')
PROC=os.path.join(BASE,'data/processed'); RAW=os.path.join(BASE,'data/raw')
SORTIE=os.path.join(BASE,'resultats/comparaison'); os.makedirs(SORTIE,exist_ok=True)
MALADIES=['Pleural Effusion','Edema','Cardiomegaly','Atelectasis','Pneumonia']
device='cuda' if torch.cuda.is_available() else 'cpu'
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
torch.manual_seed(0); np.random.seed(0); random.seed(0)

# ---------- 1. DONNEES (identiques a notre modele) ----------
print('Chargement des donnees...', flush=True)
pf16=os.path.join(PROC,'image_patches_f16.npy')
patches=np.load(pf16,mmap_mode='r') if os.path.exists(pf16) else np.load(os.path.join(PROC,'image_patches_clean.npy'),mmap_mode='r')
index=pd.read_csv(os.path.join(PROC,'image_index_clean.csv'))
chex=pd.read_csv(os.path.join(RAW,'labels_detection.csv'))
for m in MALADIES:
    lab=dict(zip(chex['study_id'],(chex[m]==1.0).astype(int)))
    index[m]=index['study_id'].map(lab)
index=index.dropna(subset=MALADIES).reset_index(drop=True)
for m in MALADIES: index[m]=index[m].astype(int)

# EHR encode (Bio_ClinicalBERT) -> on moyenne en un vecteur 768 par patient
from transformers import AutoTokenizer, AutoModel
tok=AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
bert=AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT').to(device).eval()
ehr=pd.read_csv(os.path.join(RAW,'cohort_ehr_text_enriched.csv'))
txt_of=dict(zip(ehr['subject_id'],ehr['texte'])) if 'texte' in ehr.columns else dict(zip(ehr['subject_id'],ehr.iloc[:,-1]))
EMBC={}
def ehr_vec(subj):
    if subj in EMBC: return EMBC[subj]
    enc=tok(str(txt_of.get(subj,'patient')),truncation=True,max_length=256,return_tensors='pt').to(device)
    with torch.no_grad():
        v=bert(**enc).last_hidden_state[0].mean(0).cpu().numpy()   # moyenne -> 768
    EMBC[subj]=v; return v

# ---------- 2. SPLIT PAR PATIENT (identique : 70/15/15) ----------
pats=list(index['subject_id'].unique()); random.Random(42).shuffle(pats)
n=len(pats); ntr=int(0.70*n); nva=int(0.15*n)
tr=set(pats[:ntr]); va=set(pats[ntr:ntr+nva]); te=set(pats[ntr+nva:])
def split(s): return index[index['subject_id'].isin(s)].reset_index(drop=True)
d_tr,d_va,d_te=split(tr),split(va),split(te)
print(f'Train {len(d_tr)} | Val {len(d_va)} | Test {len(d_te)} images', flush=True)

# ---------- 3. FEATURES (memes entrees que notre modele) ----------
def img_vec(i): return np.asarray(patches[i]).mean(0)   # 196x768 -> 768
row_of={str(d):i for i,d in enumerate(index['dicom_id'])}
def batch(df):
    X_img,X_ehr,Y=[],[],[]
    for _,r in df.iterrows():
        dic=str(r['dicom_id'])
        if dic not in row_of: continue
        X_img.append(img_vec(row_of[dic]))
        X_ehr.append(ehr_vec(int(r['subject_id'])))
        Y.append([int(r[m]) for m in MALADIES])
    return (torch.tensor(np.array(X_img),dtype=torch.float32),
            torch.tensor(np.array(X_ehr),dtype=torch.float32),
            torch.tensor(np.array(Y),dtype=torch.float32))
print('Extraction des features...', flush=True)
Xi_tr,Xe_tr,Y_tr=batch(d_tr); Xi_va,Xe_va,Y_va=batch(d_va); Xi_te,Xe_te,Y_te=batch(d_te)

# ---------- 4. FUSION MedFuse (LSTM sequence-de-2) ----------
# Reproduit models/fusion.py forward_lstm_fused : [ehr_feat, cxr_feat] -> LSTM -> cls
class MedFuseLSTM(nn.Module):
    def __init__(self, dim=768, hidden=256, n=5):
        super().__init__()
        self.proj_img=nn.Linear(768,dim)     # aligne les deux modalites
        self.proj_ehr=nn.Linear(768,dim)
        self.lstm=nn.LSTM(dim,hidden,batch_first=True)   # LSTM de fusion (coeur MedFuse)
        self.cls=nn.Sequential(nn.Linear(hidden,n))
    def forward(self,x_img,x_ehr):
        a=self.proj_ehr(x_ehr)[:,None,:]     # (B,1,dim)
        b=self.proj_img(x_img)[:,None,:]     # (B,1,dim)
        seq=torch.cat([a,b],dim=1)           # (B,2,dim) -> sequence de 2 modalites
        _,(ht,_)=self.lstm(seq)
        return self.cls(ht.squeeze(0))

model=MedFuseLSTM().to(device)
opt=torch.optim.Adam(model.parameters(),lr=1e-3)
lossf=nn.BCEWithLogitsLoss()

def evaluate(Xi,Xe,Y):
    model.eval()
    with torch.no_grad():
        p=torch.sigmoid(model(Xi.to(device),Xe.to(device))).cpu().numpy()
    y=Y.numpy()
    au=[roc_auc_score(y[:,k],p[:,k]) for k in range(len(MALADIES)) if y[:,k].sum()>0]
    ap=[average_precision_score(y[:,k],p[:,k]) for k in range(len(MALADIES)) if y[:,k].sum()>0]
    f1=f1_score(y,(p>0.5).astype(int),average='macro',zero_division=0)
    return np.mean(au),np.mean(ap),f1,p,y

# ---------- 5. ENTRAINEMENT ----------
print('\nEntrainement MedFuse-LSTM sur notre cohort...', flush=True)
B=256; best=0; best_state=None
for ep in range(30):
    model.train(); perm=torch.randperm(len(Xi_tr))
    for i in range(0,len(Xi_tr),B):
        idx=perm[i:i+B]
        opt.zero_grad()
        out=model(Xi_tr[idx].to(device),Xe_tr[idx].to(device))
        loss=lossf(out,Y_tr[idx].to(device)); loss.backward(); opt.step()
    au,ap,f1,_,_=evaluate(Xi_va,Xe_va,Y_va)
    if au>best: best=au; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
    if ep%5==0: print(f'  epoch {ep:2d}  val AUROC={au:.3f}  AUPRC={ap:.3f}', flush=True)

# ---------- 6. EVALUATION FINALE (test) ----------
model.load_state_dict(best_state)
au,ap,f1,p,y=evaluate(Xi_te,Xe_te,Y_te)
print('\n=== MedFuse-LSTM sur NOTRE cohort (test 15%) ===', flush=True)
print(f'  AUROC macro : {au:.3f}', flush=True)
print(f'  AUPRC macro : {ap:.3f}', flush=True)
print(f'  F1 macro    : {f1:.3f}', flush=True)
print('\n  Par pathologie (AUROC / AUPRC):', flush=True)
for k,m in enumerate(MALADIES):
    if y[:,k].sum()>0:
        print(f'    {m:18s} {roc_auc_score(y[:,k],p[:,k]):.3f} / {average_precision_score(y[:,k],p[:,k]):.3f}', flush=True)
print('\n  >>> A comparer a NOTRE modele : AUROC 0.808 / AUPRC 0.401', flush=True)

import json
json.dump({'model':'MedFuse-LSTM (our cohort)','AUROC':float(au),'AUPRC':float(ap),'F1':float(f1),
           'n_test':int(len(Y_te))}, open(os.path.join(SORTIE,'medfuse_notre_cohort.json'),'w'),indent=2)
print(f'\nSauvegarde : {os.path.join(SORTIE,"medfuse_notre_cohort.json")}', flush=True)
