#!/usr/bin/env python3
# =====================================================================
# 13_recherche_Lconsist_variantes.py
# RECHERCHE APPROFONDIE SUR L_consist (priorite #1).
#
# Compare 5 formulations de la coherence cross-modale :
#   0. Baseline            : pas de L_consist
#   1. MSE                 : notre version actuelle
#   2. JSD                 : divergence Jensen-Shannon (comme DrFuse)
#   3. KL symetrique       : KL(img||txt) + KL(txt||img)
#   4. MSE ponderee confiance : penalise plus quand les 2 sont incertains
#
# + INTERVALLES DE CONFIANCE 95% par bootstrap (1000 iterations),
#   comme DrFuse -> validite statistique.
# + graphes de comparaison.
# =====================================================================

import os, json, random, argparse
import numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score
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
parser.add_argument('--ehr', default='enrichi')
parser.add_argument('--lambda_c', type=float, default=2.0)
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--boot', type=int, default=1000)   # iterations bootstrap
args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('=== RECHERCHE L_consist : VARIANTES + BOOTSTRAP ===', flush=True)
print('Maladie:', args.maladie, '| lambda:', args.lambda_c, '| device:', device, flush=True)

# ---------- donnees ----------
pf16=os.path.join(PROC,'image_patches_f16.npy'); pf32=os.path.join(PROC,'image_patches_clean.npy')
patches = np.load(pf16, mmap_mode='r') if os.path.exists(pf16) else np.load(pf32, mmap_mode='r')
index=pd.read_csv(os.path.join(PROC,'image_index_clean.csv'))
row_of={d:i for i,d in enumerate(index['dicom_id'])}
ehr_file='cohort_ehr_text_enriched.csv' if args.ehr=='enrichi' else None
ehr=pd.read_csv(os.path.join(RAW,ehr_file)) if ehr_file and os.path.exists(os.path.join(RAW,ehr_file)) else pd.read_csv(os.path.join(PROC,'cohort_ehr_text_clean.csv'))
chex=pd.read_csv(os.path.join(RAW,'labels_detection.csv')); chex['label']=(chex[args.maladie]==1.0).astype(int)
labels=dict(zip(chex['study_id'],chex['label']))
index['label']=index['study_id'].map(labels); index=index.dropna(subset=['label']).reset_index(drop=True)
index['label']=index['label'].astype(int)

random.seed(42); pats=list(index['subject_id'].unique()); random.shuffle(pats)
n=len(pats); a=int(0.70*n); b=int(0.85*n)
p_tr,p_va,p_te=set(pats[:a]),set(pats[a:b]),set(pats[b:])
def mk(ps):
    s=index[index.subject_id.isin(ps)]
    return [(row_of[r['dicom_id']],r['subject_id'],int(r['label'])) for _,r in s.iterrows() if r['dicom_id'] in row_of]
train_s,val_s,test_s=mk(p_tr),mk(p_va),mk(p_te)
print(f'train {len(train_s)} / val {len(val_s)} / test {len(test_s)}', flush=True)

from transformers import AutoTokenizer, AutoModel
TXT='emilyalsentzer/Bio_ClinicalBERT'
tok=AutoTokenizer.from_pretrained(TXT); bert=AutoModel.from_pretrained(TXT).eval().to(device)
DIM=bert.config.hidden_size; text_emb={}
for subj in index['subject_id'].unique():
    sub=ehr[ehr.subject_id==subj]; txt=sub['texte'].iloc[0] if len(sub) else 'patient .'
    inp=tok(txt,return_tensors='pt',truncation=True,max_length=256).to(device)
    with torch.no_grad(): out=bert(**inp)
    text_emb[subj]=out.last_hidden_state[0].cpu()
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

# ---------- les differentes formulations de coherence ----------
def perte_coherence(variante, pi_logit, pt_logit):
    pi=torch.sigmoid(pi_logit); pt=torch.sigmoid(pt_logit); eps=1e-7
    if variante=='MSE':
        return F.mse_loss(pi,pt)
    if variante=='JSD':
        # Jensen-Shannon sur distributions de Bernoulli [p,1-p]
        Pi=torch.stack([pi,1-pi],dim=-1).clamp(eps,1); Pt=torch.stack([pt,1-pt],dim=-1).clamp(eps,1)
        M=0.5*(Pi+Pt)
        kl=lambda a,b:(a*(a.log()-b.log())).sum(-1)
        return (0.5*kl(Pi,M)+0.5*kl(Pt,M)).mean()
    if variante=='KL_sym':
        Pi=torch.stack([pi,1-pi],dim=-1).clamp(eps,1); Pt=torch.stack([pt,1-pt],dim=-1).clamp(eps,1)
        kl=lambda a,b:(a*(a.log()-b.log())).sum(-1)
        return (kl(Pi,Pt)+kl(Pt,Pi)).mean()
    if variante=='MSE_conf':
        # penalise davantage quand les DEUX modalites sont incertaines (proches de 0.5)
        poids=(1-(pi-0.5).abs())*(1-(pt-0.5).abs())
        return (poids*(pi-pt)**2).mean()
    return torch.tensor(0.0,device=pi_logit.device)

def entrainer(variante, lam):
    torch.manual_seed(0); model=Modele(DIM).to(device)
    npos=sum(y for *_,y in train_s); nneg=len(train_s)-npos
    pw=torch.tensor([max(nneg,1)/max(npos,1)]).to(device); bce=nn.BCEWithLogitsLoss(pos_weight=pw)
    opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    best=1e9; bs=None; wait=0
    for ep in range(args.epochs):
        random.shuffle(train_s); model.train()
        for (i,subj,y) in train_s:
            pi=get(i).unsqueeze(0).to(device); te=text_emb[subj].unsqueeze(0).to(device)
            yt=torch.tensor([float(y)]).to(device); fus,img,txt=model(pi,te)
            loss=bce(fus,yt)+bce(img,yt)+bce(txt,yt)
            if variante!='Baseline' and lam>0: loss=loss+lam*perte_coherence(variante,img,txt)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); lv=0
        with torch.no_grad():
            for (i,subj,y) in val_s:
                pi=get(i).unsqueeze(0).to(device); te=text_emb[subj].unsqueeze(0).to(device)
                lv+=bce(model(pi,te)[0],torch.tensor([float(y)]).to(device)).item()
        lv/=max(len(val_s),1)
        if lv<best: best=lv; bs={k:v.clone() for k,v in model.state_dict().items()}; wait=0
        else:
            wait+=1
            if wait>=6: break
    model.load_state_dict(bs); model.eval()
    yt,yp,pimg,ptxt=[],[],[],[]
    with torch.no_grad():
        for (i,subj,y) in test_s:
            pi=get(i).unsqueeze(0).to(device); te=text_emb[subj].unsqueeze(0).to(device)
            fus,img,txt=model(pi,te)
            yt.append(y); yp.append(torch.sigmoid(fus).item())
            pimg.append(torch.sigmoid(img).item()); ptxt.append(torch.sigmoid(txt).item())
    return np.array(yt),np.array(yp),np.array(pimg),np.array(ptxt)

# ---------- BOOTSTRAP : intervalle de confiance 95% ----------
def bootstrap_ic(yt, yp, metrique, n_boot=1000):
    rng=np.random.default_rng(42); scores=[]
    N=len(yt)
    for _ in range(n_boot):
        idx=rng.integers(0,N,N)
        if len(np.unique(yt[idx]))<2: continue
        try: scores.append(metrique(yt[idx],yp[idx]))
        except Exception: pass
    scores=np.array(scores)
    return float(np.mean(scores)), float(np.percentile(scores,2.5)), float(np.percentile(scores,97.5))

# ---------- LANCER TOUTES LES VARIANTES ----------
VARIANTES=['Baseline','MSE','JSD','KL_sym','MSE_conf']
resultats={}
for v in VARIANTES:
    yt,yp,pimg,ptxt=entrainer(v, args.lambda_c)
    au_m,au_lo,au_hi=bootstrap_ic(yt,yp,roc_auc_score,args.boot)
    ap_m,ap_lo,ap_hi=bootstrap_ic(yt,yp,average_precision_score,args.boot)
    desaccord=float(np.mean(np.abs(pimg-ptxt)))
    resultats[v]={'AUROC':au_m,'AUROC_lo':au_lo,'AUROC_hi':au_hi,
                  'AUPRC':ap_m,'AUPRC_lo':ap_lo,'AUPRC_hi':ap_hi,'Desaccord':desaccord}
    print(f'  {v:10s}: AUROC={au_m:.3f} [{au_lo:.3f}, {au_hi:.3f}]  AUPRC={ap_m:.3f} [{ap_lo:.3f}, {ap_hi:.3f}]  desaccord={desaccord:.3f}', flush=True)

# ---------- sauvegarde tableau ----------
tag=f'{args.maladie.replace(" ","_")}_{args.ehr}'
df=pd.DataFrame(resultats).T.round(4)
df.to_csv(os.path.join(RES,f'variantes_Lconsist_{tag}.csv'))

# ---------- GRAPHE 1 : AUROC avec barres d erreur (IC 95%) ----------
noms=list(resultats.keys())
au=[resultats[v]['AUROC'] for v in noms]
err_lo=[resultats[v]['AUROC']-resultats[v]['AUROC_lo'] for v in noms]
err_hi=[resultats[v]['AUROC_hi']-resultats[v]['AUROC'] for v in noms]
couleurs=['#9AA7B2','#0E7C86','#B07D2B','#7E6BB0','#2E7D5B']
plt.figure(figsize=(9,5.5))
plt.bar(noms,au,yerr=[err_lo,err_hi],capsize=6,color=couleurs,edgecolor='white')
plt.ylabel('AUROC (IC 95%)'); plt.ylim(min(au)-0.08,min(1.0,max(au)+0.06))
plt.title(f'Variantes de L_consist — {args.maladie} (bootstrap {args.boot})')
plt.grid(axis='y',alpha=0.3)
for i,v in enumerate(au): plt.text(i,v+0.005,f'{v:.3f}',ha='center',fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(RES,f'variantes_AUROC_{tag}.png'),dpi=150,bbox_inches='tight')

# ---------- GRAPHE 2 : desaccord image-texte par variante ----------
des=[resultats[v]['Desaccord'] for v in noms]
plt.figure(figsize=(9,5))
plt.bar(noms,des,color=couleurs,edgecolor='white')
plt.ylabel('Desaccord image-texte (plus bas = plus coherent)')
plt.title(f'Coherence cross-modale par variante — {args.maladie}')
plt.grid(axis='y',alpha=0.3)
for i,v in enumerate(des): plt.text(i,v+0.002,f'{v:.3f}',ha='center',fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(RES,f'variantes_desaccord_{tag}.png'),dpi=150,bbox_inches='tight')

with open(os.path.join(RES,f'variantes_Lconsist_{tag}.json'),'w') as f:
    json.dump(resultats,f,indent=2)

# meilleure variante par AUPRC
best=max(noms,key=lambda v:resultats[v]['AUPRC'])
print(f'\n>>> Meilleure variante (AUPRC) : {best}', flush=True)
print('=== TERMINE ===', flush=True)
print('Fichiers : variantes_Lconsist_'+tag+'.csv + 2 graphes (AUROC IC95, desaccord)', flush=True)
