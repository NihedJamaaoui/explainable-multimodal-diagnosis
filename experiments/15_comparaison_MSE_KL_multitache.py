#!/usr/bin/env python3
# =====================================================================
# 15_comparaison_MSE_KL_multitache.py
# COMPARAISON DIRECTE MSE vs KL_sym sur les 8 TACHES (grande cohorte).
# Pour chaque tache : Baseline / L_consist-MSE / L_consist-KL_sym
# Metriques : AUROC, AUPRC + desaccord image-texte
# Graphes : comparaison directe MSE vs KL par tache + desaccord.
# =====================================================================

import os, json, random
import numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=os.path.expanduser('~/scratch/MultimodalVLM')
PROC=os.path.join(BASE,'data/processed'); RAW=os.path.join(BASE,'data/raw'); RES=os.path.join(BASE,'resultats')
os.makedirs(RES,exist_ok=True)
TXT_ID='emilyalsentzer/Bio_ClinicalBERT'; LAMBDA=2.0; EPOCHS=40
device='cuda' if torch.cuda.is_available() else 'cpu'
C_BASE='#9AA7B2'; C_MSE='#0E7C86'; C_KL='#B07D2B'

print('=== COMPARAISON MSE vs KL_sym — 8 TACHES ===', flush=True)
pf16=os.path.join(PROC,'image_patches_f16.npy')
patches=np.load(pf16,mmap_mode='r') if os.path.exists(pf16) else np.load(os.path.join(PROC,'image_patches_clean.npy'),mmap_mode='r')
index0=pd.read_csv(os.path.join(PROC,'image_index_clean.csv'))
row_of={d:i for i,d in enumerate(index0['dicom_id'])}
ehr=pd.read_csv(os.path.join(RAW,'cohort_ehr_text_enriched.csv'))
chex=pd.read_csv(os.path.join(RAW,'labels_detection.csv')); outcome=pd.read_csv(os.path.join(RAW,'labels_outcome.csv'))

MALADIES=['Pleural Effusion','Edema','Cardiomegaly','Atelectasis','Pneumonia']
def lab_det(m):
    c=chex.copy(); c['label']=(c[m]==1.0).astype(int); return dict(zip(c['study_id'],c['label'])),'study'
def lab_out(col,seuil=None):
    o=outcome.dropna(subset=[col]).copy()
    o['label']=(o[col]>seuil).astype(int) if seuil is not None else (o[col]==1).astype(int)
    return o.groupby('subject_id')['label'].max().to_dict(),'subject'

from transformers import AutoTokenizer, AutoModel
tok=AutoTokenizer.from_pretrained(TXT_ID); bert=AutoModel.from_pretrained(TXT_ID).eval().to(device)
DIM=bert.config.hidden_size; EMB={}
for subj in index0['subject_id'].unique():
    sub=ehr[ehr.subject_id==subj]; txt=sub['texte'].iloc[0] if len(sub) else 'patient .'
    inp=tok(txt,return_tensors='pt',truncation=True,max_length=256).to(device)
    with torch.no_grad(): out=bert(**inp)
    EMB[subj]=out.last_hidden_state[0].cpu()
del bert; torch.cuda.empty_cache(); print('Texte encode.', flush=True)

class Modele(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.proj=nn.Linear(768,dim); self.cross=nn.MultiheadAttention(dim,8,batch_first=True)
        self.h_img=nn.Linear(dim,1); self.h_txt=nn.Linear(dim,1); self.h_fus=nn.Linear(dim,1)
    def forward(self,p,t):
        img=self.proj(p); fus,_=self.cross(t,img,img)
        return self.h_fus(fus.mean(1)).squeeze(-1),self.h_img(img.mean(1)).squeeze(-1),self.h_txt(t.mean(1)).squeeze(-1)

def get(i): return torch.tensor(np.asarray(patches[i]),dtype=torch.float32)

def coherence(variante,img_logit,txt_logit):
    pi=torch.sigmoid(img_logit); pt=torch.sigmoid(txt_logit); eps=1e-7
    if variante=='MSE':
        return F.mse_loss(pi,pt)
    if variante=='KL_sym':
        Pi=torch.stack([pi,1-pi],dim=-1).clamp(eps,1); Pt=torch.stack([pt,1-pt],dim=-1).clamp(eps,1)
        kl=lambda a,b:(a*(a.log()-b.log())).sum(-1)
        return (kl(Pi,Pt)+kl(Pt,Pi)).mean()
    return torch.tensor(0.0,device=img_logit.device)

def prep(lm,niv):
    idx=index0.copy(); idx['label']=idx['study_id'].map(lm) if niv=='study' else idx['subject_id'].map(lm)
    idx=idx.dropna(subset=['label']).reset_index(drop=True); idx['label']=idx['label'].astype(int)
    random.seed(42); pats=list(idx['subject_id'].unique()); random.shuffle(pats)
    n=len(pats); a=int(.7*n); b=int(.85*n)
    def mk(ps):
        s=idx[idx.subject_id.isin(ps)]
        return [(row_of[r['dicom_id']],r['subject_id'],int(r['label'])) for _,r in s.iterrows() if r['dicom_id'] in row_of]
    return mk(set(pats[:a])),mk(set(pats[a:b])),mk(set(pats[b:]))

def entrainer(tr,va,te,variante):
    torch.manual_seed(0); model=Modele(DIM).to(device)
    npos=sum(y for *_,y in tr); nneg=len(tr)-npos
    pw=torch.tensor([max(nneg,1)/max(npos,1)]).to(device); bce=nn.BCEWithLogitsLoss(pos_weight=pw)
    opt=torch.optim.Adam(model.parameters(),lr=1e-4); best=1e9; bs=None; wait=0
    for ep in range(EPOCHS):
        random.shuffle(tr); model.train()
        for (i,subj,y) in tr:
            pi=get(i).unsqueeze(0).to(device); te2=EMB[subj].unsqueeze(0).to(device)
            yt=torch.tensor([float(y)]).to(device); fus,img,txt=model(pi,te2)
            loss=bce(fus,yt)+bce(img,yt)+bce(txt,yt)
            if variante!='Baseline': loss=loss+LAMBDA*coherence(variante,img,txt)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); lv=0
        with torch.no_grad():
            for (i,subj,y) in va:
                pi=get(i).unsqueeze(0).to(device); te2=EMB[subj].unsqueeze(0).to(device)
                lv+=bce(model(pi,te2)[0],torch.tensor([float(y)]).to(device)).item()
        lv/=max(len(va),1)
        if lv<best: best=lv; bs={k:v.clone() for k,v in model.state_dict().items()}; wait=0
        else:
            wait+=1
            if wait>=6: break
    model.load_state_dict(bs); model.eval(); yt,yp,pimg,ptxt=[],[],[],[]
    with torch.no_grad():
        for (i,subj,y) in te:
            pi=get(i).unsqueeze(0).to(device); te2=EMB[subj].unsqueeze(0).to(device)
            fus,img,txt=model(pi,te2)
            yt.append(y); yp.append(torch.sigmoid(fus).item())
            pimg.append(torch.sigmoid(img).item()); ptxt.append(torch.sigmoid(txt).item())
    yp2=[1 if x>.5 else 0 for x in yp]
    return {'AUROC':roc_auc_score(yt,yp),'AUPRC':average_precision_score(yt,yp),
            'Recall':recall_score(yt,yp2,zero_division=0),
            'Desaccord':float(np.mean(np.abs(np.array(pimg)-np.array(ptxt))))}

TACHES=[(f'Détection: {m}',*lab_det(m)) for m in MALADIES]
TACHES+=[('Évolution: Mortalité',*lab_out('hospital_expire_flag')),
         ('Évolution: Séjour long (>7j)',*lab_out('duree_sejour_jours',7)),
         ('Évolution: Réadmission 30j',*lab_out('readmission_30j'))]

res=[]
for nom,lm,niv in TACHES:
    tr,va,te=prep(lm,niv); npos=sum(y for *_,y in te)
    if npos<5 or len(te)<20:
        print(f'[SKIP] {nom}', flush=True); continue
    mb=entrainer(tr,va,te,'Baseline')
    mm=entrainer(tr,va,te,'MSE')
    mk=entrainer(tr,va,te,'KL_sym')
    res.append({'Tache':nom,**{f'base_{k}':v for k,v in mb.items()},
                **{f'MSE_{k}':v for k,v in mm.items()},**{f'KL_{k}':v for k,v in mk.items()}})
    print(f'  {nom:30s} AUROC base={mb["AUROC"]:.3f} MSE={mm["AUROC"]:.3f} KL={mk["AUROC"]:.3f} | desacc MSE={mm["Desaccord"]:.3f} KL={mk["Desaccord"]:.3f}', flush=True)

df=pd.DataFrame(res); df.round(4).to_csv(os.path.join(RES,'comparaison_MSE_KL_8taches.csv'),index=False)
t=[x['Tache'].split(': ')[1] for x in res]; y=np.arange(len(t))

# GRAPHE 1 : AUROC — base vs MSE vs KL par tache
plt.figure(figsize=(11,6)); h=0.26
plt.barh(y-h,[x['base_AUROC'] for x in res],h,label='Baseline',color=C_BASE)
plt.barh(y,[x['MSE_AUROC'] for x in res],h,label='L_consist MSE',color=C_MSE)
plt.barh(y+h,[x['KL_AUROC'] for x in res],h,label='L_consist KL_sym',color=C_KL)
plt.yticks(y,t,fontsize=9); plt.xlabel('AUROC'); plt.xlim(0.5,1)
plt.title('MSE vs KL_sym — AUROC par tache (grande cohorte)')
plt.legend(); plt.grid(axis='x',alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(RES,'comp_MSE_KL_AUROC.png'),dpi=150,bbox_inches='tight')

# GRAPHE 2 : DESACCORD — MSE vs KL par tache (le point cle)
plt.figure(figsize=(11,6)); h=0.35
plt.barh(y-h/2,[x['MSE_Desaccord'] for x in res],h,label='MSE',color=C_MSE)
plt.barh(y+h/2,[x['KL_Desaccord'] for x in res],h,label='KL_sym',color=C_KL)
plt.yticks(y,t,fontsize=9); plt.xlabel('Desaccord image-texte (plus bas = plus coherent)')
plt.title('MSE vs KL_sym — coherence cross-modale par tache')
plt.legend(); plt.grid(axis='x',alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(RES,'comp_MSE_KL_desaccord.png'),dpi=150,bbox_inches='tight')

# synthese
gain_auroc_kl=np.mean([x['KL_AUROC']-x['MSE_AUROC'] for x in res])
reduc_desaccord=np.mean([x['MSE_Desaccord']-x['KL_Desaccord'] for x in res])
print(f'\n>>> KL vs MSE : diff AUROC moyenne = {gain_auroc_kl:+.4f}', flush=True)
print(f'>>> KL reduit le desaccord de {reduc_desaccord:+.4f} en moyenne', flush=True)
print('=== TERMINE ===', flush=True)
print('Fichiers : comparaison_MSE_KL_8taches.csv, comp_MSE_KL_AUROC.png, comp_MSE_KL_desaccord.png', flush=True)
