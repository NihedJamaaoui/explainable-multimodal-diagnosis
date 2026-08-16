#!/usr/bin/env python3
# =====================================================================
# 25_multilabel_variantes_seeds.py
# AMELIORATION L_consist EN MULTI-LABEL.
# Deux nouveautes par rapport au script 24 :
#   1. on teste KL_sym (jamais teste en multi-label) en plus de MSE
#   2. on repete chaque condition sur 3 GRAINES differentes
#      -> moyenne +/- ecart-type, pour separer le VRAI effet du BRUIT
#
# Conditions : Baseline / L_consist-MSE / L_consist-KL_sym
# Metriques  : AUROC moyen, F1 macro, Hamming loss, desaccord image-texte
# =====================================================================

import os, json, random
import numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, hamming_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=os.path.expanduser('~/scratch/MultimodalVLM')
PROC=os.path.join(BASE,'data/processed'); RAW=os.path.join(BASE,'data/raw'); RES=os.path.join(BASE,'resultats')
os.makedirs(RES,exist_ok=True)
MALADIES=['Pleural Effusion','Edema','Cardiomegaly','Atelectasis','Pneumonia']
NOMS={'Pleural Effusion':'Épanchement','Edema':'Œdème','Cardiomegaly':'Cardiomégalie',
      'Atelectasis':'Atélectasie','Pneumonia':'Pneumonie'}
LAMBDA=2.0; EPOCHS=30; GRAINES=[0,1,2]
CONDITIONS=['Baseline','MSE','KL_sym']
device='cuda' if torch.cuda.is_available() else 'cpu'
print('=== MULTI-LABEL : variantes L_consist x 3 graines ===', flush=True)

pf16=os.path.join(PROC,'image_patches_f16.npy')
patches=np.load(pf16,mmap_mode='r') if os.path.exists(pf16) else np.load(os.path.join(PROC,'image_patches_clean.npy'),mmap_mode='r')
index=pd.read_csv(os.path.join(PROC,'image_index_clean.csv'))
row_of={d:i for i,d in enumerate(index['dicom_id'])}
ehr=pd.read_csv(os.path.join(RAW,'cohort_ehr_text_enriched.csv'))
chex=pd.read_csv(os.path.join(RAW,'labels_detection.csv'))
for m in MALADIES:
    lab=dict(zip(chex['study_id'],(chex[m]==1.0).astype(int)))
    index[m]=index['study_id'].map(lab)
index=index.dropna(subset=MALADIES).reset_index(drop=True)
for m in MALADIES: index[m]=index[m].astype(int)

# separation par patient : FIXE (graine 42) pour que toutes les conditions
# soient comparees sur exactement les memes donnees
random.seed(42); pats=list(index['subject_id'].unique()); random.shuffle(pats)
n=len(pats); a=int(.7*n); b=int(.85*n)
def mk(ps):
    s=index[index.subject_id.isin(ps)]
    return [(row_of[r['dicom_id']],r['subject_id'],[int(r[m]) for m in MALADIES]) for _,r in s.iterrows() if r['dicom_id'] in row_of]
train_s,val_s,test_s=mk(set(pats[:a])),mk(set(pats[a:b])),mk(set(pats[b:]))
print(f'train {len(train_s)} / val {len(val_s)} / test {len(test_s)}', flush=True)

from transformers import AutoTokenizer, AutoModel
TXT='emilyalsentzer/Bio_ClinicalBERT'
tok=AutoTokenizer.from_pretrained(TXT); bert=AutoModel.from_pretrained(TXT).eval().to(device)
DIM=bert.config.hidden_size; EMB={}
for subj in index['subject_id'].unique():
    sub=ehr[ehr.subject_id==subj]; txt=sub['texte'].iloc[0] if len(sub) else 'patient .'
    inp=tok(txt,return_tensors='pt',truncation=True,max_length=256).to(device)
    with torch.no_grad(): out=bert(**inp)
    EMB[subj]=out.last_hidden_state[0].cpu()
del bert; torch.cuda.empty_cache(); print('Texte encode.', flush=True)

class Modele(nn.Module):
    def __init__(self,dim,n):
        super().__init__()
        self.proj=nn.Linear(768,dim); self.cross=nn.MultiheadAttention(dim,8,batch_first=True)
        self.h_img=nn.Linear(dim,n); self.h_txt=nn.Linear(dim,n); self.h_fus=nn.Linear(dim,n)
    def forward(self,p,t):
        img=self.proj(p); fus,_=self.cross(t,img,img)
        return self.h_fus(fus.mean(1)),self.h_img(img.mean(1)),self.h_txt(t.mean(1))

def get(i): return torch.tensor(np.asarray(patches[i]),dtype=torch.float32)
Y=np.array([y for *_,y in train_s]); pw=[]
for k in range(len(MALADIES)):
    npos=Y[:,k].sum(); pw.append(max(len(Y)-npos,1)/max(npos,1))
pw=torch.tensor(pw,dtype=torch.float32).to(device)

def coherence(cond,img_logit,txt_logit):
    """Distance entre les predictions image et texte, selon la variante."""
    pi=torch.sigmoid(img_logit); pt=torch.sigmoid(txt_logit); eps=1e-7
    if cond=='MSE':
        return F.mse_loss(pi,pt)
    if cond=='KL_sym':
        Pi=torch.stack([pi,1-pi],dim=-1).clamp(eps,1); Pt=torch.stack([pt,1-pt],dim=-1).clamp(eps,1)
        kl=lambda x,y:(x*(x.log()-y.log())).sum(-1)
        return (kl(Pi,Pt)+kl(Pt,Pi)).mean()
    return torch.tensor(0.0,device=img_logit.device)

def seuils_optimaux(Pval,Yval):
    seuils=[]
    for k in range(len(MALADIES)):
        best_f1,best_s=0,0.5
        for s in np.arange(0.1,0.9,0.05):
            f=f1_score(Yval[:,k],(Pval[:,k]>s).astype(int),zero_division=0)
            if f>best_f1: best_f1,best_s=f,s
        seuils.append(best_s)
    return np.array(seuils)

def un_run(cond,graine):
    torch.manual_seed(graine); np.random.seed(graine)
    model=Modele(DIM,len(MALADIES)).to(device)
    bce=nn.BCEWithLogitsLoss(pos_weight=pw); opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    rng=random.Random(graine)
    best=1e9; bs=None; wait=0
    for ep in range(EPOCHS):
        rng.shuffle(train_s); model.train()
        for (i,subj,y) in train_s:
            pi=get(i).unsqueeze(0).to(device); te=EMB[subj].unsqueeze(0).to(device)
            yt=torch.tensor([y],dtype=torch.float32).to(device); fus,img,txt=model(pi,te)
            loss=bce(fus,yt)+bce(img,yt)+bce(txt,yt)
            if cond!='Baseline': loss=loss+LAMBDA*coherence(cond,img,txt)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); lv=0
        with torch.no_grad():
            for (i,subj,y) in val_s:
                pi=get(i).unsqueeze(0).to(device); te=EMB[subj].unsqueeze(0).to(device)
                lv+=bce(model(pi,te)[0],torch.tensor([y],dtype=torch.float32).to(device)).item()
        lv/=max(len(val_s),1)
        if lv<best: best=lv; bs={k:v.clone() for k,v in model.state_dict().items()}; wait=0
        else:
            wait+=1
            if wait>=5: break
    model.load_state_dict(bs); model.eval()
    def collecter(S):
        P,Yv,Di=[],[],[]
        with torch.no_grad():
            for (i,subj,y) in S:
                pi=get(i).unsqueeze(0).to(device); te=EMB[subj].unsqueeze(0).to(device)
                fus,img,txt=model(pi,te)
                P.append(torch.sigmoid(fus).squeeze(0).cpu().numpy()); Yv.append(y)
                Di.append(float(np.mean(np.abs(torch.sigmoid(img).squeeze(0).cpu().numpy()
                                              -torch.sigmoid(txt).squeeze(0).cpu().numpy()))))
        return np.array(P),np.array(Yv),float(np.mean(Di))
    Pv,Yv,_=collecter(val_s); Pt,Yt,desac=collecter(test_s)
    seuils=seuils_optimaux(Pv,Yv)
    aur=[roc_auc_score(Yt[:,k],Pt[:,k]) if len(np.unique(Yt[:,k]))>1 else np.nan for k in range(len(MALADIES))]
    preds=np.stack([(Pt[:,k]>seuils[k]).astype(int) for k in range(len(MALADIES))],axis=1)
    return {'AUROC_moyen':float(np.nanmean(aur)),
            'F1_macro':f1_score(Yt,preds,average='macro',zero_division=0),
            'Hamming':hamming_loss(Yt,preds),
            'Desaccord':desac,
            'AUROC_par_maladie':aur}

# ---------------- LANCER TOUTES LES CONDITIONS x GRAINES ----------------
brut=[]
for cond in CONDITIONS:
    for g in GRAINES:
        m=un_run(cond,g)
        brut.append({'condition':cond,'graine':g,**{k:v for k,v in m.items() if k!='AUROC_par_maladie'}})
        print(f'  {cond:9s} graine {g} : AUROC={m["AUROC_moyen"]:.3f} F1={m["F1_macro"]:.3f} Hamming={m["Hamming"]:.3f} desacc={m["Desaccord"]:.3f}', flush=True)

df=pd.DataFrame(brut); df.round(4).to_csv(os.path.join(RES,'multilabel_seeds_brut.csv'),index=False)

# ---------------- MOYENNE +/- ECART-TYPE ----------------
mets=['AUROC_moyen','F1_macro','Hamming','Desaccord']
moy=df.groupby('condition')[mets].mean().round(4)
std=df.groupby('condition')[mets].std().round(4)
synth=pd.DataFrame({m:[f'{moy.loc[c,m]:.3f} ± {std.loc[c,m]:.3f}' for c in CONDITIONS] for m in mets},index=CONDITIONS)
synth.to_csv(os.path.join(RES,'multilabel_seeds_synthese.csv'))
print('\n=== MOYENNE ± ECART-TYPE (3 graines) ===', flush=True)
print(synth.to_string(), flush=True)

# ---------------- INTERPRETATION AUTOMATIQUE ----------------
print('\n=== LECTURE ===', flush=True)
for m in ['F1_macro','Desaccord']:
    base_m=moy.loc['Baseline',m]; base_s=std.loc['Baseline',m]
    for c in ['MSE','KL_sym']:
        ecart=moy.loc[c,m]-base_m
        bruit=max(base_s,std.loc[c,m])
        verdict='EFFET REEL' if abs(ecart)>bruit else 'dans le bruit'
        print(f'  {m:10s} {c:7s} vs Baseline : {ecart:+.4f}  (bruit ±{bruit:.4f}) -> {verdict}', flush=True)

# ---------------- GRAPHES ----------------
x=np.arange(len(CONDITIONS)); COL=['#9AA7B2','#0E7C86','#B07D2B']
fig,axes=plt.subplots(1,3,figsize=(15,4.5))
for ax,m,titre in zip(axes,['F1_macro','Hamming','Desaccord'],
                      ['F1 macro (haut=mieux)','Hamming loss (bas=mieux)','Desaccord image-texte (bas=mieux)']):
    ax.bar(x,[moy.loc[c,m] for c in CONDITIONS],yerr=[std.loc[c,m] for c in CONDITIONS],
           capsize=6,color=COL,edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels(CONDITIONS,fontsize=10); ax.set_title(titre,fontsize=11)
    ax.grid(axis='y',alpha=0.3)
    for k,c in enumerate(CONDITIONS):
        ax.text(k,moy.loc[c,m],f'{moy.loc[c,m]:.3f}',ha='center',va='bottom',fontsize=9)
plt.suptitle('Multi-label : Baseline vs MSE vs KL_sym (moyenne ± ecart-type, 3 graines)',fontsize=12)
plt.tight_layout(); plt.savefig(os.path.join(RES,'multilabel_seeds_comparaison.png'),dpi=150,bbox_inches='tight')

print('\n=== TERMINE ===', flush=True)
print('Fichiers : multilabel_seeds_brut.csv, multilabel_seeds_synthese.csv,', flush=True)
print('           multilabel_seeds_comparaison.png', flush=True)
