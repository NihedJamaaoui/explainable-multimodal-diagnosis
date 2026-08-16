#!/usr/bin/env python3
# =====================================================================
# comparer_fusions.py
# COMPARAISON NIVEAU 1 : 3 fusions de reference sur NOTRE cohort exact.
#   - Early  : concatenation des features (early fusion)
#   - Late   : moyenne des predictions des deux tetes (late fusion)
#   - DrFuse : alignement latent (perte JSD entre les deux branches)
#
# Memes entrees que notre modele (patches BiomedCLIP moyennes + EHR
# Bio_ClinicalBERT moyenne), meme split 70/15/15 par patient, memes 5
# pathologies. Seule l'architecture de fusion change.
#
# Sortie : AUROC / AUPRC macro par fusion -> complete le tableau
#          (MedFuse 0.754/0.334 ; notre cross-attention 0.808/0.401).
# =====================================================================

import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, random, json
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

BASE=os.path.expanduser('~/scratch/MultimodalVLM')
PROC=os.path.join(BASE,'data/processed'); RAW=os.path.join(BASE,'data/raw')
SORTIE=os.path.join(BASE,'resultats/comparaison'); os.makedirs(SORTIE,exist_ok=True)
MALADIES=['Pleural Effusion','Edema','Cardiomegaly','Atelectasis','Pneumonia']
device='cuda' if torch.cuda.is_available() else 'cpu'
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
torch.manual_seed(0); np.random.seed(0); random.seed(0)

# ---------- donnees (identiques a MedFuse) ----------
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

from transformers import AutoTokenizer, AutoModel
tok=AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
bert=AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT').to(device).eval()
ehr=pd.read_csv(os.path.join(RAW,'cohort_ehr_text_enriched.csv'))
txt_of=dict(zip(ehr['subject_id'],ehr['texte'])) if 'texte' in ehr.columns else dict(zip(ehr['subject_id'],ehr.iloc[:,-1]))
EMBC={}
def ehr_vec(subj):
    if subj in EMBC: return EMBC[subj]
    enc=tok(str(txt_of.get(subj,'patient')),truncation=True,max_length=256,return_tensors='pt').to(device)
    with torch.no_grad(): v=bert(**enc).last_hidden_state[0].mean(0).cpu().numpy()
    EMBC[subj]=v; return v

pats=list(index['subject_id'].unique()); random.Random(42).shuffle(pats)
n=len(pats); ntr=int(0.70*n); nva=int(0.15*n)
tr=set(pats[:ntr]); va=set(pats[ntr:ntr+nva]); te=set(pats[ntr+nva:])
def split(s): return index[index['subject_id'].isin(s)].reset_index(drop=True)
d_tr,d_va,d_te=split(tr),split(va),split(te)
print(f'Train {len(d_tr)} | Val {len(d_va)} | Test {len(d_te)}', flush=True)

def img_vec(i): return np.asarray(patches[i]).mean(0)
row_of={str(d):i for i,d in enumerate(index['dicom_id'])}
def batch(df):
    Xi,Xe,Y=[],[],[]
    for _,r in df.iterrows():
        dic=str(r['dicom_id'])
        if dic not in row_of: continue
        Xi.append(img_vec(row_of[dic])); Xe.append(ehr_vec(int(r['subject_id'])))
        Y.append([int(r[m]) for m in MALADIES])
    return (torch.tensor(np.array(Xi),dtype=torch.float32),
            torch.tensor(np.array(Xe),dtype=torch.float32),
            torch.tensor(np.array(Y),dtype=torch.float32))
print('Extraction des features...', flush=True)
Xi_tr,Xe_tr,Y_tr=batch(d_tr); Xi_va,Xe_va,Y_va=batch(d_va); Xi_te,Xe_te,Y_te=batch(d_te)

# ---------- 3 fusions ----------
class Early(nn.Module):          # concatenation
    def __init__(self,d=768,h=256,n=5):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(2*d,h),nn.ReLU(),nn.Dropout(0.2),nn.Linear(h,n))
    def forward(self,xi,xe): return self.net(torch.cat([xi,xe],dim=1)), None

class Late(nn.Module):           # moyenne des predictions
    def __init__(self,d=768,h=256,n=5):
        super().__init__()
        self.img=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Linear(h,n))
        self.ehr=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Linear(h,n))
    def forward(self,xi,xe): return (self.img(xi)+self.ehr(xe))/2, None

class DrFuse(nn.Module):         # alignement latent (JSD entre les 2 branches)
    def __init__(self,d=768,h=256,n=5):
        super().__init__()
        self.enc_img=nn.Sequential(nn.Linear(d,h),nn.ReLU())
        self.enc_ehr=nn.Sequential(nn.Linear(d,h),nn.ReLU())
        self.head_img=nn.Linear(h,n); self.head_ehr=nn.Linear(h,n)
        self.fuse=nn.Linear(2*h,n)
    def forward(self,xi,xe):
        hi=self.enc_img(xi); he=self.enc_ehr(xe)
        pred=self.fuse(torch.cat([hi,he],dim=1))
        # JSD entre les distributions des deux branches (alignement latent)
        pi=torch.sigmoid(self.head_img(hi)); pe=torch.sigmoid(self.head_ehr(he))
        m=0.5*(pi+pe)+1e-8
        jsd=0.5*(pi*torch.log((pi+1e-8)/m)).mean()+0.5*(pe*torch.log((pe+1e-8)/m)).mean()
        return pred, jsd

def evaluate(model,Xi,Xe,Y):
    model.eval()
    with torch.no_grad():
        out,_=model(Xi.to(device),Xe.to(device)); p=torch.sigmoid(out).cpu().numpy()
    y=Y.numpy()
    au=np.mean([roc_auc_score(y[:,k],p[:,k]) for k in range(len(MALADIES)) if y[:,k].sum()>0])
    ap=np.mean([average_precision_score(y[:,k],p[:,k]) for k in range(len(MALADIES)) if y[:,k].sum()>0])
    f1=f1_score(y,(p>0.5).astype(int),average='macro',zero_division=0)
    return au,ap,f1,p,y

def entrainer(model,lam_align=0.0):
    model=model.to(device); opt=torch.optim.Adam(model.parameters(),lr=1e-3); lossf=nn.BCEWithLogitsLoss()
    B=256; best=0; best_state=None
    for ep in range(30):
        model.train(); perm=torch.randperm(len(Xi_tr))
        for i in range(0,len(Xi_tr),B):
            idx=perm[i:i+B]; opt.zero_grad()
            out,align=model(Xi_tr[idx].to(device),Xe_tr[idx].to(device))
            loss=lossf(out,Y_tr[idx].to(device))
            if align is not None: loss=loss+lam_align*align
            loss.backward(); opt.step()
        au,_,_,_,_=evaluate(model,Xi_va,Xe_va,Y_va)
        if au>best: best=au; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state); return model

# ---------- run ----------
resultats={'MedFuse-LSTM':{'AUROC':0.754,'AUPRC':0.334},
           'Cross-attention (ours)':{'AUROC':0.808,'AUPRC':0.401}}
configs=[('Early (concat)',Early(),0.0),('Late (average)',Late(),0.0),('DrFuse (latent align)',DrFuse(),0.5)]
for nom,mdl,lam in configs:
    print(f'\n=== {nom} ===', flush=True)
    m=entrainer(mdl,lam_align=lam)
    au,ap,f1,p,y=evaluate(m,Xi_te,Xe_te,Y_te)
    print(f'  AUROC {au:.3f} | AUPRC {ap:.3f} | F1 {f1:.3f}', flush=True)
    resultats[nom]={'AUROC':round(float(au),3),'AUPRC':round(float(ap),3),'F1':round(float(f1),3)}

print('\n\n======== TABLEAU FINAL (notre cohort, test 15%) ========', flush=True)
print(f'{"Fusion":28s} {"AUROC":>7s} {"AUPRC":>7s}', flush=True)
ordre=['MedFuse-LSTM','Early (concat)','Late (average)','DrFuse (latent align)','Cross-attention (ours)']
for k in ordre:
    r=resultats[k]; print(f'{k:28s} {r["AUROC"]:>7.3f} {r["AUPRC"]:>7.3f}', flush=True)
json.dump(resultats,open(os.path.join(SORTIE,'comparaison_fusions.json'),'w'),indent=2)
print(f'\nSauvegarde : {os.path.join(SORTIE,"comparaison_fusions.json")}', flush=True)
