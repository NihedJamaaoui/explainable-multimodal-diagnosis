#!/usr/bin/env python3
# =====================================================================
# 30_iou_mscxr.py
# VALIDATION QUANTITATIVE DE L EXPLICABILITE (IoU contre MS-CXR).
#
# Question posee : la zone que le modele regarde correspond-elle a la
# region que des radiologues ont delimitee ?
#
# Deux cartes de localisation comparees :
#   - attention  : poids d attention croisee texte -> 196 regions image
#   - gradcam    : gradient du score par rapport aux regions image
#
# Deux metriques :
#   - IoU           : recouvrement entre boite predite et boite experte
#   - pointing game : le point le plus regarde tombe-t-il dans la boite ?
#
# Trois conditions comparees : Baseline / L_consist-MSE / L_consist-KL_sym
# -> permet de repondre a : la coherence cross-modale ameliore-t-elle
#    aussi l ancrage anatomique ?
#
# IMPORTANT — geometrie : BiomedCLIP applique un redimensionnement du
# petit cote a 224 puis un rognage central. Les coordonnees des boites
# subissent donc la meme transformation avant d etre comparees.
# =====================================================================

import os, sys, json, random
import numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE=os.path.expanduser('~/scratch/MultimodalVLM')
PROC=os.path.join(BASE,'data/processed'); RAW=os.path.join(BASE,'data/raw')
JPG=os.path.join(BASE,'data/jpg'); RES=os.path.join(BASE,'resultats')
EXP=os.path.join(RES,'iou_mscxr'); os.makedirs(EXP,exist_ok=True)

MALADIES=['Pleural Effusion','Edema','Cardiomegaly','Atelectasis','Pneumonia']
NOMS={'Pleural Effusion':'Épanchement','Edema':'Œdème','Cardiomegaly':'Cardiomégalie',
      'Atelectasis':'Atélectasie','Pneumonia':'Pneumonie'}
LAMBDA=2.0; EPOCHS=30; SEUIL_CAM=0.5; N_EXEMPLES=6
import argparse
_ap=argparse.ArgumentParser()
_ap.add_argument('--conditions', default='Baseline,MSE,KL_sym',
                 help='conditions a traiter, separees par des virgules (ex: --conditions KL_sym)')
_args=_ap.parse_args()
CONDITIONS=[c.strip() for c in _args.conditions.split(',') if c.strip()]
device='cuda' if torch.cuda.is_available() else 'cpu'
print('=== VALIDATION IoU CONTRE MS-CXR ===', flush=True)
print('Conditions a traiter lors de cette execution :', CONDITIONS, flush=True)

# =====================================================================
# 1) Lecture de MS-CXR avec detection automatique des colonnes
# =====================================================================
candidats=[os.path.join(RAW,f) for f in os.listdir(RAW) if 'ms_cxr' in f.lower() or 'ms-cxr' in f.lower()]
candidats=[c for c in candidats if c.endswith('.csv') or c.endswith('.csv.gz')]
if not candidats:
    print("ERREUR : aucun fichier MS-CXR trouve dans", RAW, flush=True)
    print("Fichiers presents :", [f for f in os.listdir(RAW) if f.endswith('.csv')][:20], flush=True)
    sys.exit(1)
ms_path=candidats[0]
ms=pd.read_csv(ms_path)
print('Fichier MS-CXR :', os.path.basename(ms_path), '|', len(ms), 'annotations', flush=True)
print('Colonnes :', list(ms.columns), flush=True)

def trouver(colonnes, *noms):
    """Retourne le premier nom de colonne present (insensible a la casse)."""
    bas={c.lower():c for c in colonnes}
    for n in noms:
        if n.lower() in bas: return bas[n.lower()]
    return None

c_dicom=trouver(ms.columns,'dicom_id','dicom','image_id')
c_cat  =trouver(ms.columns,'category_name','category','label','pathology')
c_x=trouver(ms.columns,'x','bbox_x','x1'); c_y=trouver(ms.columns,'y','bbox_y','y1')
c_w=trouver(ms.columns,'w','width','bbox_w'); c_h=trouver(ms.columns,'h','height','bbox_h')
c_iw=trouver(ms.columns,'image_width','img_width'); c_ih=trouver(ms.columns,'image_height','img_height')
if not all([c_dicom,c_x,c_y,c_w,c_h]):
    print('ERREUR : colonnes de boite introuvables. Colonnes disponibles ci-dessus.', flush=True)
    sys.exit(1)
print(f'Colonnes retenues -> dicom:{c_dicom} cat:{c_cat} box:({c_x},{c_y},{c_w},{c_h}) taille:({c_iw},{c_ih})', flush=True)

# =====================================================================
# 2) Geometrie : boite en pixels d origine -> coordonnees 0-1 sur la
#    grille 224x224 vue par le modele (resize petit cote + crop central)
# =====================================================================
def boite_vers_grille(x,y,w,h,W,H):
    """Applique resize(petit cote=224) + centre-crop(224), puis normalise."""
    if not W or not H or W<=0 or H<=0:
        return None
    s=224.0/min(W,H)
    nW,nH=W*s,H*s
    dx,dy=(nW-224.0)/2.0,(nH-224.0)/2.0
    x1=x*s-dx; y1=y*s-dy; x2=(x+w)*s-dx; y2=(y+h)*s-dy
    # rognage aux bords de l image vue par le modele
    x1=max(0.0,min(224.0,x1)); x2=max(0.0,min(224.0,x2))
    y1=max(0.0,min(224.0,y1)); y2=max(0.0,min(224.0,y2))
    if x2-x1<4 or y2-y1<4:      # boite quasi entierement rognee
        return None
    return [x1/224.0, y1/224.0, x2/224.0, y2/224.0]

def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    aa=(a[2]-a[0])*(a[3]-a[1]); ab=(b[2]-b[0])*(b[3]-b[1])
    u=aa+ab-inter
    return inter/u if u>0 else 0.0

def carte_vers_boite(carte14, seuil=SEUIL_CAM):
    """Carte 14x14 -> boite englobante de la zone au-dessus du seuil."""
    c=carte14.astype(np.float64)
    c=(c-c.min())/(c.max()-c.min()+1e-8)
    ys,xs=np.where(c>=seuil)
    if len(xs)==0: return None
    return [xs.min()/14.0, ys.min()/14.0, (xs.max()+1)/14.0, (ys.max()+1)/14.0]

def point_max(carte14):
    """Coordonnees 0-1 du point le plus regarde (centre de la case max)."""
    k=int(np.argmax(carte14)); r,c=divmod(k,14)
    return ((c+0.5)/14.0,(r+0.5)/14.0)

def dans_boite(pt,b):
    return b[0]<=pt[0]<=b[2] and b[1]<=pt[1]<=b[3]

# =====================================================================
# 3) Donnees du projet
# =====================================================================
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

sujet_de={r['dicom_id']:r['subject_id'] for _,r in index.iterrows()}

# ---- Strategie de decoupage ----
# MS-CXR est petit (~1000 images annotees). Filtrer sur un decoupage de test
# classique ne laisserait qu une poignee d annotations. On procede autrement :
# tous les patients porteurs d une annotation MS-CXR sont EXCLUS de
# l entrainement, et l evaluation porte sur la TOTALITE de leurs annotations.
# Aucune fuite de donnees, effectif d evaluation maximise.
dicoms_ms=set(str(x) for x in ms[c_dicom].astype(str))
p_ms={sujet_de[d] for d in dicoms_ms if d in sujet_de}
print('Patients porteurs d une annotation MS-CXR presents dans la cohorte :', len(p_ms), flush=True)

pats_dispo=[p for p in index['subject_id'].unique() if p not in p_ms]
random.seed(42); random.shuffle(pats_dispo)
a=int(.85*len(pats_dispo))
p_tr,p_va=set(pats_dispo[:a]),set(pats_dispo[a:])
def mk(ps):
    s=index[index.subject_id.isin(ps)]
    return [(row_of[r['dicom_id']],r['subject_id'],[int(r[m]) for m in MALADIES]) for _,r in s.iterrows() if r['dicom_id'] in row_of]
train_s,val_s=mk(p_tr),mk(p_va)
print(f'train {len(train_s)} / val {len(val_s)}  (patients MS-CXR exclus de l entrainement)', flush=True)

# ---- annotations MS-CXR exploitables : toutes celles dont l image est presente ----
# ---- annotations MS-CXR : regroupement des boites d une meme constatation ----
# Une constatation bilaterale est delimitee par plusieurs boites (ex. « bibasilar
# opacities » -> base gauche + base droite). Le modele ne produisant qu une zone,
# on fusionne ces boites en une region englobante, convention usuelle en ancrage
# de phrases. On ecarte par ailleurs les categories absentes de nos 5 pathologies,
# faute de tete de decision correspondante.
c_txt=trouver(ms.columns,'label_text','phrase','sentence')
brut={}
ignorees={'image_absente':0,'boite_rognee':0,'categorie_hors_liste':0}
for _,r in ms.iterrows():
    dic=str(r[c_dicom])
    if dic not in row_of:
        ignorees['image_absente']+=1; continue
    cat=str(r[c_cat]) if c_cat else ''
    if cat not in MALADIES:
        ignorees['categorie_hors_liste']+=1; continue
    W=r[c_iw] if c_iw else None; H=r[c_ih] if c_ih else None
    if W is None or H is None or (isinstance(W,float) and np.isnan(W)):
        chemin=os.path.join(JPG,f'{dic}.jpg')
        if os.path.exists(chemin):
            im=Image.open(chemin); W,H=im.size
        else:
            ignorees['image_absente']+=1; continue
    bg=boite_vers_grille(float(r[c_x]),float(r[c_y]),float(r[c_w]),float(r[c_h]),float(W),float(H))
    if bg is None:
        ignorees['boite_rognee']+=1; continue
    phr=str(r[c_txt]) if c_txt else ''
    brut.setdefault((dic,cat,phr),[]).append(bg)

annots=[]
for (dic,cat,phr),boites in brut.items():
    union=[min(b[0] for b in boites), min(b[1] for b in boites),
           max(b[2] for b in boites), max(b[3] for b in boites)]
    annots.append({'dicom_id':dic,'ligne':row_of[dic],'boite':union,
                   'categorie':cat,'k':MALADIES.index(cat),'n_boites':len(boites)})
print(f'Constatations regroupees : {len(annots)}  (a partir de {sum(len(v) for v in brut.values())} boites)', flush=True)

print(f'Annotations exploitables : {len(annots)}', flush=True)
print('  ignorees ->', ignorees, flush=True)
if len(annots)<10:
    print('ATTENTION : trop peu d annotations pour conclure. Verifier l appariement des dicom_id.', flush=True)

# =====================================================================
# 4) Modele multi-label (identique aux experiences precedentes)
# =====================================================================
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
    def forward(self,p,t,retour_attn=False):
        img=self.proj(p)
        fus,attn=self.cross(t,img,img,need_weights=retour_attn,average_attn_weights=True)
        out=(self.h_fus(fus.mean(1)),self.h_img(img.mean(1)),self.h_txt(t.mean(1)))
        return (out,attn,img) if retour_attn else out

def get(i): return torch.tensor(np.asarray(patches[i]),dtype=torch.float32)
Y=np.array([y for *_,y in train_s]); pw=[]
for k in range(len(MALADIES)):
    npos=Y[:,k].sum(); pw.append(max(len(Y)-npos,1)/max(npos,1))
pw=torch.tensor(pw,dtype=torch.float32).to(device)

def coherence(cond,img_logit,txt_logit):
    pi=torch.sigmoid(img_logit); pt=torch.sigmoid(txt_logit); eps=1e-7
    if cond=='MSE': return F.mse_loss(pi,pt)
    if cond=='KL_sym':
        Pi=torch.stack([pi,1-pi],dim=-1).clamp(eps,1); Pt=torch.stack([pt,1-pt],dim=-1).clamp(eps,1)
        kl=lambda x,y:(x*(x.log()-y.log())).sum(-1)
        return (kl(Pi,Pt)+kl(Pt,Pi)).mean()
    return torch.tensor(0.0,device=img_logit.device)

def entrainer(cond):
    torch.manual_seed(0); model=Modele(DIM,len(MALADIES)).to(device)
    bce=nn.BCEWithLogitsLoss(pos_weight=pw); opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    best=1e9; bs=None; wait=0
    for ep in range(EPOCHS):
        random.shuffle(train_s); model.train()
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
    model.load_state_dict(bs); model.eval(); return model

# =====================================================================
# 5) Cartes de localisation
# =====================================================================
def carte_attention(model,i,subj):
    pi=get(i).unsqueeze(0).to(device); te=EMB[subj].unsqueeze(0).to(device)
    with torch.no_grad():
        (fus,_,_),attn,_=model(pi,te,retour_attn=True)
    return attn.mean(1).squeeze(0).cpu().numpy().reshape(14,14)

def carte_gradcam(model,i,subj,k):
    """Grad-CAM sur les jetons image : gradient du score de la maladie k."""
    pi=get(i).unsqueeze(0).to(device).requires_grad_(True)
    te=EMB[subj].unsqueeze(0).to(device)
    fus,_,_=model(pi,te)
    score=fus[0,k]
    model.zero_grad()
    if pi.grad is not None: pi.grad=None
    score.backward()
    g=pi.grad.detach().squeeze(0)              # (196,768)
    a=pi.detach().squeeze(0)                   # (196,768)
    cam=torch.relu((a*g).sum(-1)).cpu().numpy()
    if cam.max()<=0: cam=np.abs((a*g).sum(-1).cpu().numpy())
    return cam.reshape(14,14)

# =====================================================================
# 6) Evaluation de l ancrage pour chaque condition
# =====================================================================
CUMUL=os.path.join(EXP,'iou_resultats_cumul.json')
resultats=json.load(open(CUMUL)) if os.path.exists(CUMUL) else []
# si une condition est relancee, on remplace ses anciennes lignes
resultats=[r for r in resultats if r['condition'] not in CONDITIONS]
if resultats:
    print('Resultats deja presents pour :', sorted({r['condition'] for r in resultats}), flush=True)
detail=[]
modeles={}
for cond in CONDITIONS:
    print(f'\n--- Entrainement {cond} ---', flush=True)
    model=entrainer(cond); modeles[cond]=model
    for methode in ['attention','gradcam']:
        ious=[]; hits=[]; par_cat={}
        for an in annots:
            i=an['ligne']; subj=sujet_de[an['dicom_id']]
            k=an['k']
            carte=carte_attention(model,i,subj) if methode=='attention' else carte_gradcam(model,i,subj,k)
            bp=carte_vers_boite(carte)
            if bp is None: continue
            v=iou(bp,an['boite']); h=1.0 if dans_boite(point_max(carte),an['boite']) else 0.0
            ious.append(v); hits.append(h)
            par_cat.setdefault(an['categorie'],[]).append(v)
            if cond=='KL_sym' and methode=='gradcam':
                detail.append({'dicom_id':an['dicom_id'],'categorie':an['categorie'],
                               'IoU':round(v,3),'pointing':int(h)})
        moy_cat={c:round(float(np.mean(v)),3) for c,v in par_cat.items() if len(v)>=3}
        resultats.append({'condition':cond,'methode':methode,'n':len(ious),
                          'IoU_moyen':round(float(np.mean(ious)),4) if ious else 0,
                          'IoU_median':round(float(np.median(ious)),4) if ious else 0,
                          'pointing_game':round(float(np.mean(hits)),4) if hits else 0,
                          'par_categorie':moy_cat})
        print(f'  {methode:9s}: IoU moyen={np.mean(ious):.3f}  median={np.median(ious):.3f}  '
              f'pointing={np.mean(hits):.3f}  (n={len(ious)})', flush=True)
    # sauvegarde immediate : une condition terminee n est jamais reperdue
    with open(CUMUL,'w') as f: json.dump(resultats,f,indent=2)
    print(f'  -> condition {cond} sauvegardee dans {os.path.basename(CUMUL)}', flush=True)

with open(CUMUL,'w') as f: json.dump(resultats,f,indent=2)

ORDRE=['Baseline','MSE','KL_sym']
presentes=[c for c in ORDRE if any(r['condition']==c for r in resultats)]
df=pd.DataFrame([{k:v for k,v in r.items() if k!='par_categorie'} for r in resultats])
df.to_csv(os.path.join(EXP,'iou_synthese.csv'),index=False)
if detail: pd.DataFrame(detail).to_csv(os.path.join(EXP,'iou_detail_KLsym_gradcam.csv'),index=False)
with open(os.path.join(EXP,'iou_complet.json'),'w') as f: json.dump(resultats,f,indent=2)

print('\n=== SYNTHESE (toutes conditions disponibles) ===', flush=True)
print(df.to_string(index=False), flush=True)
if len(presentes)<len(ORDRE):
    manquantes=[c for c in ORDRE if c not in presentes]
    print('\nConditions encore manquantes :', manquantes, flush=True)
    print('Relancer avec : python 30_iou_mscxr.py --conditions', ','.join(manquantes), flush=True)

# ---- IoU par pathologie, meilleure configuration ----
best=max(resultats,key=lambda r:r['IoU_moyen'])
print(f"\nMeilleure configuration : {best['condition']} / {best['methode']}", flush=True)
print('IoU par pathologie :', best['par_categorie'], flush=True)

# =====================================================================
# 7) Graphes
# =====================================================================
C={'Baseline':'#9AA7B2','MSE':'#0E7C86','KL_sym':'#B07D2B'}
fig,axes=plt.subplots(1,2,figsize=(13,5))
for ax,met,titre in zip(axes,['IoU_moyen','pointing_game'],
                        ['IoU moyen (zone regardee vs experts)','Pointing game (taux de reussite)']):
    largeur=0.35; x=np.arange(len(presentes))
    for j,methode in enumerate(['attention','gradcam']):
        vals=[next((r[met] for r in resultats if r['condition']==c and r['methode']==methode),0) for c in presentes]
        ax.bar(x+(j-0.5)*largeur,vals,largeur,label=methode,
               color=['#0E7C86','#B07D2B'][j],edgecolor='white')
        for xi,v in zip(x+(j-0.5)*largeur,vals): ax.text(xi,v,f'{v:.3f}',ha='center',va='bottom',fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(presentes); ax.set_title(titre,fontsize=11)
    ax.grid(axis='y',alpha=0.3); ax.legend()
plt.suptitle('Ancrage anatomique : validation contre les delimitations MS-CXR',fontsize=12)
plt.tight_layout(); plt.savefig(os.path.join(EXP,'iou_comparaison.png'),dpi=150,bbox_inches='tight')

# ---- exemples visuels : boite experte vs zone regardee ----
if modeles:
    nom_modele='KL_sym' if 'KL_sym' in modeles else list(modeles)[0]
    model=modeles[nom_modele]
    print(f'\nExemples visuels generes avec le modele : {nom_modele}', flush=True)
    choisis=annots[:N_EXEMPLES]
else:
    model=None; choisis=[]
for idx,an in enumerate(choisis):
    i=an['ligne']; subj=sujet_de[an['dicom_id']]
    k=an['k']
    carte=carte_gradcam(model,i,subj,k)
    bp=carte_vers_boite(carte)
    chemin=os.path.join(JPG,f"{an['dicom_id']}.jpg")
    if not os.path.exists(chemin): continue
    im=Image.open(chemin).convert('L')
    s=224.0/min(im.size); im=im.resize((int(im.size[0]*s),int(im.size[1]*s)))
    l=(im.size[0]-224)//2; t=(im.size[1]-224)//2
    im=im.crop((l,t,l+224,t+224))
    fig,ax=plt.subplots(figsize=(5.5,5.5))
    ax.imshow(np.array(im),cmap='gray')
    ax.imshow(np.kron(carte,np.ones((16,16))),cmap='jet',alpha=0.4)
    gx1,gy1,gx2,gy2=[c*224 for c in an['boite']]
    ax.add_patch(mpatches.Rectangle((gx1,gy1),gx2-gx1,gy2-gy1,fill=False,edgecolor='#2E7D5B',lw=2.5,label='expert'))
    if bp:
        px1,py1,px2,py2=[c*224 for c in bp]
        ax.add_patch(mpatches.Rectangle((px1,py1),px2-px1,py2-py1,fill=False,edgecolor='#B0413E',lw=2.5,ls='--',label='modele'))
        ax.set_title(f"{an['categorie']} — IoU = {iou(bp,an['boite']):.2f}",fontsize=11)
    ax.legend(loc='lower right',fontsize=9); ax.axis('off')
    plt.tight_layout(); plt.savefig(os.path.join(EXP,f'ancrage_{idx+1}.png'),dpi=130,bbox_inches='tight')
    plt.close()

print('\n=== TERMINE ===', flush=True)
print('Fichiers dans', EXP, flush=True)
print('  iou_synthese.csv, iou_detail_KLsym_gradcam.csv, iou_comparaison.png, ancrage_N.png', flush=True)
