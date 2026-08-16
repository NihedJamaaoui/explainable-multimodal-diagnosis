#!/usr/bin/env python3
# =====================================================================
# 44_rapports_cohorte.py  (version RAPIDE — charge les poids, pas de reentrainement)
#
# Pre-requis : avoir lance 32_lground.py au moins une fois AVEC la
# sauvegarde des poids (fichier resultats/lground/poids_Lground_g0.pt).
#
# Chaine :
#   1) reconstruit le modele et charge les poids sauvegardes (~2 min)
#   2) pour N images de test : predictions + carte d attention
#   3) RAG + Llama-3.1 -> rapport diagnostique
#   4) sauvegarde rapport .txt + figure .png + JSON
#
# Job court (~30 min) : demarre vite dans la file.
# =====================================================================

import os, sys, json
import numpy as np
import torch, torch.nn as nn

BASE=os.path.expanduser('~/scratch/MultimodalVLM')
PROC=os.path.join(BASE,'data/processed'); RAW=os.path.join(BASE,'data/raw')
EXP=os.path.join(BASE,'resultats/lground')
JPG=os.path.join(BASE,'data/jpg')
SORTIE=os.path.join(BASE,'resultats/rapports'); os.makedirs(SORTIE,exist_ok=True)

MALADIES=['Pleural Effusion','Edema','Cardiomegaly','Atelectasis','Pneumonia']
device='cuda' if torch.cuda.is_available() else 'cpu'
POIDS=os.environ.get('POIDS',os.path.join(EXP,'poids_Lground_g0.pt'))
N=int(os.environ.get('N_RAPPORTS','5'))

# =====================================================================
# 1) Donnees : patches, index, EMB texte
# =====================================================================
print('Chargement des donnees...', flush=True)
pf16=os.path.join(PROC,'image_patches_f16.npy')
patches=np.load(pf16,mmap_mode='r') if os.path.exists(pf16) else np.load(os.path.join(PROC,'image_patches_clean.npy'),mmap_mode='r')
import pandas as pd
index=pd.read_csv(os.path.join(PROC,'image_index_clean.csv'))
row_of={str(d):i for i,d in enumerate(index['dicom_id'])}
chex=pd.read_csv(os.path.join(RAW,'labels_detection.csv'))
for m in MALADIES:
    lab=dict(zip(chex['study_id'],(chex[m]==1.0).astype(int)))
    index[m]=index['study_id'].map(lab)
index=index.dropna(subset=MALADIES).reset_index(drop=True)
for m in MALADIES: index[m]=index[m].astype(int)

# encodeur texte (EMB) — Bio_ClinicalBERT gele
from transformers import AutoTokenizer, AutoModel
print('Encodage du texte clinique...', flush=True)
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
tok=AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
bert=AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT').to(device).eval()
DIM=bert.config.hidden_size
ehr=pd.read_csv(os.path.join(RAW,'cohort_ehr_text_enriched.csv'))
txt_of=dict(zip(ehr['subject_id'],ehr['texte'])) if 'texte' in ehr.columns else \
        dict(zip(ehr['subject_id'],ehr.iloc[:,-1]))
EMB={}
def emb_texte(subj):
    if subj in EMB: return EMB[subj]
    t=str(txt_of.get(subj,'patient'))
    enc=tok(t,truncation=True,max_length=256,return_tensors='pt').to(device)
    with torch.no_grad():
        EMB[subj]=bert(**enc).last_hidden_state[0].cpu()
    return EMB[subj]

def get(i): return torch.tensor(np.asarray(patches[i]),dtype=torch.float32)

# =====================================================================
# 2) Reconstruire le modele + charger les poids
# =====================================================================
class Modele(nn.Module):
    def __init__(self,dim,n):
        super().__init__()
        self.proj=nn.Linear(768,dim); self.cross=nn.MultiheadAttention(dim,8,batch_first=True)
        self.h_img=nn.Linear(dim,n); self.h_txt=nn.Linear(dim,n); self.h_fus=nn.Linear(dim,n)
    def forward(self,p,t,attn_grad=False):
        img=self.proj(p)
        fus,attn=self.cross(t,img,img,need_weights=True,average_attn_weights=True)
        return (self.h_fus(fus.mean(1)),self.h_img(img.mean(1)),self.h_txt(t.mean(1)),attn)

print(f'Chargement des poids : {POIDS}', flush=True)
if not os.path.exists(POIDS):
    print('ERREUR : poids introuvables. Lance d abord 32_lground.py (il sauvegarde poids_Lground_g0.pt).', flush=True)
    sys.exit(1)
model=Modele(DIM,len(MALADIES)).to(device)
model.load_state_dict(torch.load(POIDS,map_location=device))
model.eval()
print('Modele charge.', flush=True)

def carte_attention(i,subj):
    pi=get(i).unsqueeze(0).to(device); te=emb_texte(subj).unsqueeze(0).to(device)
    with torch.no_grad(): _,_,_,attn=model(pi,te,attn_grad=True)
    return attn.mean(1).squeeze(0).cpu().numpy().reshape(14,14)

# =====================================================================
# 3) Generateur RAG + Llama (module 42)
# =====================================================================
import importlib.util
spec=importlib.util.spec_from_file_location('gen42',
      os.path.join(os.path.dirname(os.path.abspath(__file__)),'42_generer_rapport.py'))
gen42=importlib.util.module_from_spec(spec); spec.loader.exec_module(gen42)
print('\nChargement du generateur (RAG + Llama)...', flush=True)
GEN=gen42.charger_generateur()

# =====================================================================
# 4) Choisir les images : soit une liste fixe (cas demonstratifs), soit au hasard
# =====================================================================
# Pour utiliser des cas choisis a la main, colle leurs dicom_id ici
# (obtenus via 45_choisir_cas.py). Laisse la liste vide pour un choix au hasard.
DICOMS_CHOISIS = [
    'ae8c827a-2211830f-386b6460-28826c1c-999c1769',
    '82fb163a-d6d66d47-974aa54e-724bc7ba-1ac4858e',
    '7ff4e7c7-59ffae35-f3e8c04c-b3e5efc7-ba128f43',
    '3dfdaa3b-a1532010-731d6b5d-0048bac5-731cdd39',
    '81a2d3fb-10e5ed3b-8ee41b6b-dda4f09e-0e3a10ee',
]

def infos(dic):
    r=index[index['dicom_id'].astype(str)==dic]
    if len(r)==0: return None
    r=r.iloc[0]; ys=[int(r[m]) for m in MALADIES]
    return (dic,int(r['subject_id']),ys)

if DICOMS_CHOISIS:
    choisis=[infos(d) for d in DICOMS_CHOISIS if infos(d) and d in row_of]
    print(f'Cas choisis a la main : {len(choisis)}', flush=True)
else:
    candidats=[]
    for _,r in index.iterrows():
        dic=str(r['dicom_id'])
        ys=[int(r[m]) for m in MALADIES]
        if sum(ys)>0 and dic in row_of:
            candidats.append((dic,int(r['subject_id']),ys))
        if len(candidats)>=N*4: break
    import random; random.seed(0); random.shuffle(candidats)
    choisis=candidats[:N]

from PIL import Image
def img_224(dic):
    p=os.path.join(JPG,f'{dic}.jpg')
    if not os.path.exists(p): return None
    im=Image.open(p).convert('L'); W,H=im.size
    s=224/min(W,H); im=im.resize((int(W*s),int(H*s)))
    l=(im.size[0]-224)//2; t=(im.size[1]-224)//2
    return np.array(im.crop((l,t,l+224,t+224)))

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

print(f'\n=== Generation de {len(choisis)} rapports ===', flush=True)
tous=[]
for k,(dic,subj,ys) in enumerate(choisis,1):
    i=row_of[dic]
    with torch.no_grad():
        pi=get(i).unsqueeze(0).to(device); te=emb_texte(subj).unsqueeze(0).to(device)
        probs=torch.sigmoid(model(pi,te)[0]).squeeze(0).cpu().numpy()
    carte=carte_attention(i,subj); carte=np.clip(carte,0,None); carte=carte/(carte.max()+1e-8)
    r=gen42.generer_rapport(GEN,probs,carte,dic,MALADIES)
    verite=[MALADIES[j] for j in range(len(MALADIES)) if ys[j]==1]
    r['verite_terrain']=verite; tous.append(r)

    with open(os.path.join(SORTIE,f'rapport_{dic}.txt'),'w') as f:
        f.write(f"Image : {dic}\nVerite terrain : {', '.join(verite)}\n")
        f.write(f"Detecte : {r['detectees']}\nZone : {r['zone']}\n\n{r['rapport']}\n")

    base=img_224(dic)
    if base is not None:
        fig,ax=plt.subplots(1,2,figsize=(8,4.2))
        ax[0].imshow(base,cmap='gray'); ax[0].set_title('Radiographie'); ax[0].axis('off')
        ax[1].imshow(base,cmap='gray')
        att=np.kron(carte,np.ones((16,16)))[:224,:224]
        ax[1].imshow(att,cmap='jet',alpha=0.45)
        ax[1].set_title("Zone d'attention (L_ground)"); ax[1].axis('off')
        plt.tight_layout(); plt.savefig(os.path.join(SORTIE,f'figure_{dic}.png'),dpi=150,bbox_inches='tight'); plt.close()

    print(f'\n--- Rapport {k}/{len(choisis)} : {dic} ---', flush=True)
    print(f'  Verite terrain : {verite}', flush=True)
    print(f'  Detecte : {r["detectees"]}', flush=True)
    print(f'  Zone : {r["zone"]}', flush=True)
    print(f'  {r["rapport"][:220]}...', flush=True)

json.dump(tous,open(os.path.join(SORTIE,'rapports_tous.json'),'w'),indent=2,ensure_ascii=False)
print(f'\n=== TERMINE ===\n{len(tous)} rapports + figures dans {SORTIE}', flush=True)
