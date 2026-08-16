#!/usr/bin/env python3
# =====================================================================
# 23_detection_multimaladie.py
# DETECTION MULTI-MALADIES + explicabilite.
# Prend des patients AU HASARD et detecte QUELLES maladies ils ont
# (parmi 5), compare a la verite, et montre le Grad-CAM.
#
# Un seul modele a 5 sorties (une par maladie) = classification multi-label.
# =====================================================================

import os, json, random
import numpy as np, pandas as pd, torch
import torch.nn as nn
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=os.path.expanduser('~/scratch/MultimodalVLM')
PROC=os.path.join(BASE,'data/processed'); RAW=os.path.join(BASE,'data/raw')
JPG=os.path.join(BASE,'data/jpg'); RES=os.path.join(BASE,'resultats')
EXP=os.path.join(RES,'multimaladie'); os.makedirs(EXP,exist_ok=True)
MALADIES=['Pleural Effusion','Edema','Cardiomegaly','Atelectasis','Pneumonia']
NOMS_FR={'Pleural Effusion':'Épanchement pleural','Edema':'Œdème','Cardiomegaly':'Cardiomégalie',
         'Atelectasis':'Atélectasie','Pneumonia':'Pneumonie'}
LAMBDA=2.0; EPOCHS=30; N_EXEMPLES=8; SEUIL=0.5
device='cuda' if torch.cuda.is_available() else 'cpu'
print('=== DETECTION MULTI-MALADIES ===', flush=True)

pf16=os.path.join(PROC,'image_patches_f16.npy')
patches=np.load(pf16,mmap_mode='r') if os.path.exists(pf16) else np.load(os.path.join(PROC,'image_patches_clean.npy'),mmap_mode='r')
index=pd.read_csv(os.path.join(PROC,'image_index_clean.csv'))
row_of={d:i for i,d in enumerate(index['dicom_id'])}
ehr=pd.read_csv(os.path.join(RAW,'cohort_ehr_text_enriched.csv'))
chex=pd.read_csv(os.path.join(RAW,'labels_detection.csv'))

# etiquettes multi-label : un vecteur de 5 (une case par maladie) par etude
for m in MALADIES:
    lab=dict(zip(chex['study_id'],(chex[m]==1.0).astype(int)))
    index[m]=index['study_id'].map(lab)
index=index.dropna(subset=MALADIES).reset_index(drop=True)
for m in MALADIES: index[m]=index[m].astype(int)

random.seed(42); pats=list(index['subject_id'].unique()); random.shuffle(pats)
n=len(pats); a=int(.7*n); b=int(.85*n)
p_tr,p_te=set(pats[:a]),set(pats[b:])
def mk(ps):
    s=index[index.subject_id.isin(ps)]
    return [(row_of[r['dicom_id']],r['subject_id'],r['dicom_id'],[int(r[m]) for m in MALADIES])
            for _,r in s.iterrows() if r['dicom_id'] in row_of]
train_s,test_s=mk(p_tr),mk(p_te)
print(f'train {len(train_s)} / test {len(test_s)}', flush=True)

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

# modele multi-label : 5 sorties (fusion) + 5 image + 5 texte
class ModeleMulti(nn.Module):
    def __init__(self,dim,n_cls):
        super().__init__()
        self.proj=nn.Linear(768,dim); self.cross=nn.MultiheadAttention(dim,8,batch_first=True)
        self.h_img=nn.Linear(dim,n_cls); self.h_txt=nn.Linear(dim,n_cls); self.h_fus=nn.Linear(dim,n_cls)
    def forward(self,p,t,retour_attn=False):
        img=self.proj(p); fus,attn=self.cross(t,img,img,need_weights=retour_attn,average_attn_weights=True)
        out=(self.h_fus(fus.mean(1)),self.h_img(img.mean(1)),self.h_txt(t.mean(1)))
        return (out,attn) if retour_attn else out

def get(i): return torch.tensor(np.asarray(patches[i]),dtype=torch.float32)

# poids par classe (desequilibre)
Y=np.array([y for *_,y in train_s]); pw=[]
for k in range(len(MALADIES)):
    npos=Y[:,k].sum(); nneg=len(Y)-npos; pw.append(max(nneg,1)/max(npos,1))
pw=torch.tensor(pw,dtype=torch.float32).to(device)

print('Entrainement multi-maladies...', flush=True)
torch.manual_seed(0); model=ModeleMulti(DIM,len(MALADIES)).to(device)
bce=nn.BCEWithLogitsLoss(pos_weight=pw); mse=nn.MSELoss()
opt=torch.optim.Adam(model.parameters(),lr=1e-4)
for ep in range(EPOCHS):
    random.shuffle(train_s); model.train()
    for (i,subj,dic,y) in train_s:
        pi=get(i).unsqueeze(0).to(device); te=text_emb[subj].unsqueeze(0).to(device)
        yt=torch.tensor([y],dtype=torch.float32).to(device); fus,img,txt=model(pi,te)
        loss=bce(fus,yt)+bce(img,yt)+bce(txt,yt)+LAMBDA*mse(torch.sigmoid(img),torch.sigmoid(txt))
        opt.zero_grad(); loss.backward(); opt.step()
model.eval(); print('Modele entraine.', flush=True)

def charger_image(dic):
    chemin=os.path.join(JPG,f'{dic}.jpg')
    return np.array(Image.open(chemin).convert('L').resize((224,224))) if os.path.exists(chemin) else None

# ---------- patients AU HASARD ----------
random.seed(123); echantillon=random.sample(test_s,min(N_EXEMPLES,len(test_s)))
print(f'Generation pour {len(echantillon)} patients au hasard...', flush=True)
resume=[]
for idx,(i,subj,dic,y_vrai) in enumerate(echantillon):
    pi=get(i).unsqueeze(0).to(device); te=text_emb[subj].unsqueeze(0).to(device)
    with torch.no_grad():
        (fus,img,txt),attn=model(pi,te,retour_attn=True)
        probs=torch.sigmoid(fus).squeeze(0).cpu().numpy()
    detectees=[MALADIES[k] for k in range(len(MALADIES)) if probs[k]>SEUIL]
    vraies=[MALADIES[k] for k in range(len(MALADIES)) if y_vrai[k]==1]
    carte=attn.mean(1).squeeze(0).cpu().numpy().reshape(14,14)
    imgpx=charger_image(dic)

    fig=plt.figure(figsize=(15,4.5))
    # 1) image
    ax1=fig.add_subplot(1,3,1)
    if imgpx is not None: ax1.imshow(imgpx,cmap='gray')
    ax1.set_title('Radiographie'); ax1.axis('off')
    # 2) grad-cam
    ax2=fig.add_subplot(1,3,2)
    if imgpx is not None: ax2.imshow(imgpx,cmap='gray')
    ax2.imshow(np.kron(carte,np.ones((16,16))),cmap='jet',alpha=0.5)
    ax2.set_title('Grad-CAM (zone regardee)'); ax2.axis('off')
    # 3) barres de probabilite par maladie
    ax3=fig.add_subplot(1,3,3)
    cols=[]
    for k in range(len(MALADIES)):
        pred=probs[k]>SEUIL; vrai=y_vrai[k]==1
        if pred and vrai: cols.append('#2E7D5B')      # vrai positif (vert)
        elif pred and not vrai: cols.append('#B0413E')# faux positif (rouge)
        elif not pred and vrai: cols.append('#B07D2B')# manque (orange)
        else: cols.append('#9AA7B2')                  # vrai negatif (gris)
    ax3.barh([NOMS_FR[m] for m in MALADIES],probs,color=cols)
    ax3.axvline(SEUIL,color='k',ls='--',lw=0.8); ax3.set_xlim(0,1)
    ax3.set_title('Probabilite par maladie'); ax3.set_xlabel('probabilite')
    for k,v in enumerate(probs): ax3.text(v+0.02,k,f'{v:.2f}',va='center',fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(EXP,f'patient_{idx+1}.png'),dpi=130,bbox_inches='tight')
    plt.close()

    correct=set(detectees)==set(vraies)
    resume.append({'patient':idx+1,'subject_id':subj,
                   'detectees':', '.join(NOMS_FR[m] for m in detectees) or 'aucune',
                   'vraies':', '.join(NOMS_FR[m] for m in vraies) or 'aucune',
                   'correct':'OUI' if correct else 'partiel/non'})
    print(f'  Patient {idx+1}: détecté=[{", ".join(detectees) or "aucune"}] | vrai=[{", ".join(vraies) or "aucune"}] {"✓" if correct else "≈"}', flush=True)

pd.DataFrame(resume).to_csv(os.path.join(EXP,'resume_multimaladie.csv'),index=False)
print('=== TERMINE ===', flush=True)
print(f'Figures patient_N.png + resume_multimaladie.csv dans {EXP}', flush=True)
print('Legende couleurs : vert=correct, rouge=faux positif, orange=manque, gris=absent correct', flush=True)
