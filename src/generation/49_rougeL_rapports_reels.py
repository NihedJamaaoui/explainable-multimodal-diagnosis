#!/usr/bin/env python3
# =====================================================================
# 49_rougeL_rapports_reels.py
# Calcule le ROUGE-L VRAI : genere les VRAIS rapports Llama sur N images
# de test, puis les compare aux vrais rapports du radiologue (MIMIC).
#
# Difference avec 47 : ici on appelle reellement Llama (rapport complet),
# pas un rapport minimal. C'est le ROUGE-L honnete a mettre dans le memoire.
#
# Charge les poids sauvegardes (poids_Lground_g0.pt) — pas de reentrainement.
# N controlable via N_ROUGE (defaut 200 ; commencer par 50 pour tester la vitesse).
#
# Pre-requis : rouge-score (deja installe)
# =====================================================================

import os, sys, json
import numpy as np, torch, torch.nn as nn, pandas as pd

BASE=os.path.expanduser('~/scratch/MultimodalVLM')
PROC=os.path.join(BASE,'data/processed'); RAW=os.path.join(BASE,'data/raw')
EXP=os.path.join(BASE,'resultats/lground')
SORTIE=os.path.join(BASE,'resultats/rapports'); os.makedirs(SORTIE,exist_ok=True)
REPORTS_DIR=os.path.join(RAW,'mimic-cxr-reports')
MALADIES=['Pleural Effusion','Edema','Cardiomegaly','Atelectasis','Pneumonia']
device='cuda' if torch.cuda.is_available() else 'cpu'
POIDS=os.path.join(EXP,'poids_Lground_g0.pt')
N=int(os.environ.get('N_ROUGE','200'))
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'

# --- donnees ---
print('Chargement des donnees...', flush=True)
pf16=os.path.join(PROC,'image_patches_f16.npy')
patches=np.load(pf16,mmap_mode='r') if os.path.exists(pf16) else np.load(os.path.join(PROC,'image_patches_clean.npy'),mmap_mode='r')
index=pd.read_csv(os.path.join(PROC,'image_index_clean.csv'))
row_of={str(d):i for i,d in enumerate(index['dicom_id'])}
chex=pd.read_csv(os.path.join(RAW,'labels_detection.csv'))
for m in MALADIES:
    lab=dict(zip(chex['study_id'],(chex[m]==1.0).astype(int)))
    index[m]=index['study_id'].map(lab)
index=index.dropna(subset=MALADIES).reset_index(drop=True)
for m in MALADIES: index[m]=index[m].astype(int)

# meme decoupage par patient : on prend le TEST (15% derniers)
import random
random.seed(42)
pats=list(index['subject_id'].unique()); random.shuffle(pats)
ntest=int(0.15*len(pats)); test_pats=set(pats[-ntest:])
test_idx=index[index['subject_id'].isin(test_pats)].reset_index(drop=True)
# echantillon aleatoire (meme graine que 47 pour coherence)
test_idx=test_idx.sample(n=min(N,len(test_idx)),random_state=0).reset_index(drop=True)
print(f'Echantillon : {len(test_idx)} images\n', flush=True)

# --- encodeur texte ---
from transformers import AutoTokenizer, AutoModel
tok=AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
bert=AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT').to(device).eval()
DIM=bert.config.hidden_size
ehr=pd.read_csv(os.path.join(RAW,'cohort_ehr_text_enriched.csv'))
txt_of=dict(zip(ehr['subject_id'],ehr['texte'])) if 'texte' in ehr.columns else dict(zip(ehr['subject_id'],ehr.iloc[:,-1]))
EMBC={}
def emb_texte(subj):
    if subj in EMBC: return EMBC[subj]
    enc=tok(str(txt_of.get(subj,'patient')),truncation=True,max_length=256,return_tensors='pt').to(device)
    with torch.no_grad(): EMBC[subj]=bert(**enc).last_hidden_state[0].cpu()
    return EMBC[subj]
def get(i): return torch.tensor(np.asarray(patches[i]),dtype=torch.float32)

class Modele(nn.Module):
    def __init__(self,dim,n):
        super().__init__()
        self.proj=nn.Linear(768,dim); self.cross=nn.MultiheadAttention(dim,8,batch_first=True)
        self.h_img=nn.Linear(dim,n); self.h_txt=nn.Linear(dim,n); self.h_fus=nn.Linear(dim,n)
    def forward(self,p,t,attn_grad=False):
        img=self.proj(p)
        fus,attn=self.cross(t,img,img,need_weights=True,average_attn_weights=True)
        return (self.h_fus(fus.mean(1)),self.h_img(img.mean(1)),self.h_txt(t.mean(1)),attn)

model=Modele(DIM,len(MALADIES)).to(device)
model.load_state_dict(torch.load(POIDS,map_location=device)); model.eval()
print('Modele charge.\n', flush=True)

def carte_attention(i,subj):
    pi=get(i).unsqueeze(0).to(device); te=emb_texte(subj).unsqueeze(0).to(device)
    with torch.no_grad(): _,_,_,attn=model(pi,te,attn_grad=True)
    return attn.mean(1).squeeze(0).cpu().numpy().reshape(14,14)

# --- generateur RAG + Llama (module 42) ---
import importlib.util
spec=importlib.util.spec_from_file_location('gen42',
      os.path.join(os.path.dirname(os.path.abspath(__file__)),'42_generer_rapport.py'))
gen42=importlib.util.module_from_spec(spec); spec.loader.exec_module(gen42)
print('Chargement du generateur (RAG + Llama)...', flush=True)
GEN=gen42.charger_generateur()

# --- vrai rapport radiologue ---
info=index.set_index(index['dicom_id'].astype(str))
def vrai_rapport(dic):
    if dic not in info.index: return None
    r=info.loc[dic]
    if isinstance(r,pd.DataFrame): r=r.iloc[0]
    sid=str(int(r['subject_id'])); study=int(r['study_id'])
    p=os.path.join(REPORTS_DIR,'files',f'p{sid[:2]}',f'p{sid}',f's{study}.txt')
    if not os.path.exists(p): return None
    txt=open(p,encoding='utf-8',errors='ignore').read()
    bloc=''
    for cle in ('FINDINGS:','IMPRESSION:'):
        if cle in txt: bloc+=' '+txt.split(cle,1)[1].split('\n \n')[0]
    return (bloc if bloc else txt).replace('\n',' ').strip()

# --- generer les vrais rapports + ROUGE-L ---
from rouge_score import rouge_scorer
sc=rouge_scorer.RougeScorer(['rougeL'],use_stemmer=True)
scores=[]; faits=0
print(f'Generation des vrais rapports Llama (peut etre long)...\n', flush=True)
for k,(_,r) in enumerate(test_idx.iterrows(),1):
    dic=str(r['dicom_id'])
    if dic not in row_of: continue
    ref=vrai_rapport(dic)
    if not ref: continue          # pas de rapport radiologue -> on saute
    i=row_of[dic]; subj=int(r['subject_id'])
    with torch.no_grad():
        pi=get(i).unsqueeze(0).to(device); te=emb_texte(subj).unsqueeze(0).to(device)
        probs=torch.sigmoid(model(pi,te)[0]).squeeze(0).cpu().numpy()
    carte=carte_attention(i,subj); carte=np.clip(carte,0,None); carte=carte/(carte.max()+1e-8)
    rap=gen42.generer_rapport(GEN,probs,carte,dic,MALADIES)
    s=sc.score(ref,rap['rapport'])['rougeL'].fmeasure
    scores.append(s); faits+=1
    if k%20==0: print(f'  {k}/{len(test_idx)} traites, ROUGE-L courant={np.mean(scores):.3f}', flush=True)

print(f'\n=== ROUGE-L VRAI (vrais rapports Llama, {faits} images) ===', flush=True)
print(f'  ROUGE-L moyen : {np.mean(scores):.3f}', flush=True)
print(f'  ecart-type    : {np.std(scores):.3f}', flush=True)
json.dump({'rougeL':float(np.mean(scores)),'std':float(np.std(scores)),'n':faits},
          open(os.path.join(SORTIE,'rougeL_reel.json'),'w'),indent=2)
print(f'\nSauvegarde : {os.path.join(SORTIE,"rougeL_reel.json")}', flush=True)
