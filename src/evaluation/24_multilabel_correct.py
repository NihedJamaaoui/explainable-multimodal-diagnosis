#!/usr/bin/env python3
# =====================================================================
# 24_multilabel_correct.py
# EVALUATION MULTI-LABEL CORRECTE (5 maladies simultanees).
# Corrige les erreurs d evaluation :
#   - metriques adaptees : AUROC par maladie, F1 macro, Hamming loss
#   - SEUIL OPTIMAL par maladie (calcule sur la validation)
#   - ablation : image seule / texte seul / fusion
#   - comparaison : sans vs avec L_consist
# =====================================================================

import os, json, random
import numpy as np, pandas as pd, torch
import torch.nn as nn
from sklearn.metrics import (roc_auc_score, f1_score, hamming_loss,
                             precision_score, recall_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=os.path.expanduser('~/scratch/MultimodalVLM')
PROC=os.path.join(BASE,'data/processed'); RAW=os.path.join(BASE,'data/raw'); RES=os.path.join(BASE,'resultats')
os.makedirs(RES,exist_ok=True)
MALADIES=['Pleural Effusion','Edema','Cardiomegaly','Atelectasis','Pneumonia']
NOMS={'Pleural Effusion':'Épanchement','Edema':'Œdème','Cardiomegaly':'Cardiomégalie',
      'Atelectasis':'Atélectasie','Pneumonia':'Pneumonie'}
LAMBDA=2.0; EPOCHS=35
device='cuda' if torch.cuda.is_available() else 'cpu'
print('=== MULTI-LABEL CORRECT (metriques adaptees) ===', flush=True)

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

random.seed(42); pats=list(index['subject_id'].unique()); random.shuffle(pats)
n=len(pats); a=int(.7*n); b=int(.85*n)
p_tr,p_va,p_te=set(pats[:a]),set(pats[a:b]),set(pats[b:])
def mk(ps):
    s=index[index.subject_id.isin(ps)]
    return [(row_of[r['dicom_id']],r['subject_id'],[int(r[m]) for m in MALADIES]) for _,r in s.iterrows() if r['dicom_id'] in row_of]
train_s,val_s,test_s=mk(p_tr),mk(p_va),mk(p_te)
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

def entrainer(mode,lam):
    torch.manual_seed(0); model=Modele(DIM,len(MALADIES)).to(device)
    bce=nn.BCEWithLogitsLoss(pos_weight=pw); mse=nn.MSELoss()
    opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    def sortie(fus,img,txt): return {'fusion':fus,'image':img,'texte':txt}[mode]
    best=1e9; bs=None; wait=0
    for ep in range(EPOCHS):
        random.shuffle(train_s); model.train()
        for (i,subj,y) in train_s:
            pi=get(i).unsqueeze(0).to(device); te=EMB[subj].unsqueeze(0).to(device)
            yt=torch.tensor([y],dtype=torch.float32).to(device); fus,img,txt=model(pi,te)
            loss=bce(fus,yt)+bce(img,yt)+bce(txt,yt)
            if lam>0: loss=loss+lam*mse(torch.sigmoid(img),torch.sigmoid(txt))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); lv=0
        with torch.no_grad():
            for (i,subj,y) in val_s:
                pi=get(i).unsqueeze(0).to(device); te=EMB[subj].unsqueeze(0).to(device)
                lv+=bce(sortie(*model(pi,te)),torch.tensor([y],dtype=torch.float32).to(device)).item()
        lv/=max(len(val_s),1)
        if lv<best: best=lv; bs={k:v.clone() for k,v in model.state_dict().items()}; wait=0
        else:
            wait+=1
            if wait>=6: break
    model.load_state_dict(bs); model.eval()
    def collecter(S):
        P,Yv=[],[]
        with torch.no_grad():
            for (i,subj,y) in S:
                pi=get(i).unsqueeze(0).to(device); te=EMB[subj].unsqueeze(0).to(device)
                P.append(torch.sigmoid(sortie(*model(pi,te))).squeeze(0).cpu().numpy()); Yv.append(y)
        return np.array(P),np.array(Yv)
    Pv,Yvv=collecter(val_s); Pt,Ytt=collecter(test_s)
    return Pv,Yvv,Pt,Ytt

def seuils_optimaux(Pval,Yval):
    seuils=[]
    for k in range(len(MALADIES)):
        best_f1,best_s=0,0.5
        for s in np.arange(0.1,0.9,0.05):
            f=f1_score(Yval[:,k],(Pval[:,k]>s).astype(int),zero_division=0)
            if f>best_f1: best_f1,best_s=f,s
        seuils.append(best_s)
    return np.array(seuils)

def evaluer(Pt,Ytt,seuils):
    aurocs=[roc_auc_score(Ytt[:,k],Pt[:,k]) if len(np.unique(Ytt[:,k]))>1 else float('nan') for k in range(len(MALADIES))]
    preds=np.stack([(Pt[:,k]>seuils[k]).astype(int) for k in range(len(MALADIES))],axis=1)
    return {'AUROC_par_maladie':aurocs,'AUROC_moyen':float(np.nanmean(aurocs)),
            'F1_macro':f1_score(Ytt,preds,average='macro',zero_division=0),
            'Hamming':hamming_loss(Ytt,preds),
            'Precision_macro':precision_score(Ytt,preds,average='macro',zero_division=0),
            'Recall_macro':recall_score(Ytt,preds,average='macro',zero_division=0)}

# ---------- ABLATION : image / texte / fusion ----------
print('\n--- ABLATION (seuil optimal) ---', flush=True)
resultats={}
for mode in ['texte','image','fusion']:
    Pv,Yv,Pt,Yt=entrainer(mode,0.0); seuils=seuils_optimaux(Pv,Yv)
    m=evaluer(Pt,Yt,seuils); resultats[f'ablation_{mode}']=m
    print(f'  {mode:7s}: AUROC_moy={m["AUROC_moyen"]:.3f} F1_macro={m["F1_macro"]:.3f} Hamming={m["Hamming"]:.3f}', flush=True)

# ---------- L_consist : sans vs avec ----------
print('\n--- L_consist (fusion) ---', flush=True)
Pv,Yv,Pt,Yt=entrainer('fusion',0.0); s0=seuils_optimaux(Pv,Yv); m0=evaluer(Pt,Yt,s0); resultats['sans_Lconsist']=m0
Pv,Yv,Pt,Yt=entrainer('fusion',LAMBDA); s1=seuils_optimaux(Pv,Yv); m1=evaluer(Pt,Yt,s1); resultats['avec_Lconsist']=m1
print(f'  sans L_consist : F1_macro={m0["F1_macro"]:.3f} Hamming={m0["Hamming"]:.3f}', flush=True)
print(f'  avec L_consist : F1_macro={m1["F1_macro"]:.3f} Hamming={m1["Hamming"]:.3f}', flush=True)

# AUROC par maladie (avec L_consist)
print('\n--- AUROC par maladie (avec L_consist) ---', flush=True)
for k,mal in enumerate(MALADIES):
    print(f'  {NOMS[mal]:14s}: AUROC={m1["AUROC_par_maladie"][k]:.3f}', flush=True)

# ---------- sauvegarde ----------
lignes=[]
for k,mal in enumerate(MALADIES):
    lignes.append({'Maladie':NOMS[mal],'AUROC':round(m1['AUROC_par_maladie'][k],3),'seuil_optimal':round(float(s1[k]),2)})
pd.DataFrame(lignes).to_csv(os.path.join(RES,'multilabel_auroc_par_maladie.csv'),index=False)

synth=pd.DataFrame({
 'Image seule':{k:resultats['ablation_image'][k] for k in ['AUROC_moyen','F1_macro','Hamming']},
 'Texte seul':{k:resultats['ablation_texte'][k] for k in ['AUROC_moyen','F1_macro','Hamming']},
 'Fusion (baseline)':{k:m0[k] for k in ['AUROC_moyen','F1_macro','Hamming']},
 'Fusion + L_consist':{k:m1[k] for k in ['AUROC_moyen','F1_macro','Hamming']},
}).T.round(3)
synth.to_csv(os.path.join(RES,'multilabel_synthese.csv'))
print('\nSynthese :\n', synth.to_string(), flush=True)

# ---------- GRAPHE 1 : AUROC par maladie ----------
plt.figure(figsize=(9,5))
au=[m1['AUROC_par_maladie'][k] for k in range(len(MALADIES))]
plt.bar([NOMS[m] for m in MALADIES],au,color='#0E7C86')
plt.ylim(0.5,1); plt.ylabel('AUROC'); plt.title('AUROC par maladie (multi-label, +L_consist)')
plt.grid(axis='y',alpha=0.3)
for k,v in enumerate(au): plt.text(k,v+0.005,f'{v:.3f}',ha='center',fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(RES,'multilabel_AUROC_maladie.png'),dpi=150,bbox_inches='tight')

# ---------- GRAPHE 2 : ablation (F1 macro + Hamming) ----------
mods=['Image seule','Texte seul','Fusion (baseline)','Fusion + L_consist']
f1s=[resultats['ablation_image']['F1_macro'],resultats['ablation_texte']['F1_macro'],m0['F1_macro'],m1['F1_macro']]
ham=[resultats['ablation_image']['Hamming'],resultats['ablation_texte']['Hamming'],m0['Hamming'],m1['Hamming']]
x=np.arange(len(mods)); w=0.35
fig,ax=plt.subplots(figsize=(10,5))
ax.bar(x-w/2,f1s,w,label='F1 macro (haut=mieux)',color='#2E7D5B')
ax.bar(x+w/2,ham,w,label='Hamming loss (bas=mieux)',color='#B07D2B')
ax.set_xticks(x); ax.set_xticklabels(mods,fontsize=9); ax.legend(); ax.grid(axis='y',alpha=0.3)
ax.set_title('Multi-label : ablation + L_consist')
plt.tight_layout(); plt.savefig(os.path.join(RES,'multilabel_ablation.png'),dpi=150,bbox_inches='tight')

with open(os.path.join(RES,'multilabel_complet.json'),'w') as f:
    json.dump(resultats,f,indent=2)

print('\n=== TERMINE ===', flush=True)
print('Fichiers : multilabel_auroc_par_maladie.csv, multilabel_synthese.csv,', flush=True)
print('           multilabel_AUROC_maladie.png, multilabel_ablation.png', flush=True)
