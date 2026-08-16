#!/usr/bin/env python3
# =====================================================================
# 32_lground.py
# CONTRIBUTION C1 — L_ground : ancrage anatomique de l attention.
#
# Principe
# --------
# L attention du modele doit se concentrer sur les regions anatomiques ou
# la pathologie peut reellement se manifester. On construit, pour chaque
# image, un masque 14x14 a partir des boites Chest ImaGenome des regions
# pertinentes, puis on penalise l attention qui tombe en dehors :
#
#     L_ground = 1 - (attention dans le masque / attention totale)
#
# Elle vaut 0 si toute l attention est bien placee, et tend vers 1 si elle
# se disperse ailleurs.
#
#     L_total = L_tache + lambda_g * L_ground + lambda_c * L_consist
#
# Conditions comparables (argument --conditions) :
#   Baseline           : L_tache seule
#   Lconsist           : + L_consist (KL symetrique)
#   Lground            : + L_ground
#   Lground_Lconsist   : + les deux
#
# Evaluations produites :
#   - classification : AUROC par pathologie, F1 macro, Hamming
#   - ancrage sur ImaGenome (patients de test, effectif eleve)
#   - ancrage sur MS-CXR (verification independante, annotations expertes)
# =====================================================================

import os, sys, json, random, argparse
import numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, hamming_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=os.path.expanduser('~/scratch/MultimodalVLM')
PROC=os.path.join(BASE,'data/processed'); RAW=os.path.join(BASE,'data/raw')
RES=os.path.join(BASE,'resultats'); EXP=os.path.join(RES,'lground')
os.makedirs(EXP,exist_ok=True)

MALADIES=['Pleural Effusion','Edema','Cardiomegaly','Atelectasis','Pneumonia']
NOMS={'Pleural Effusion':'Épanchement','Edema':'Œdème','Cardiomegaly':'Cardiomégalie',
      'Atelectasis':'Atélectasie','Pneumonia':'Pneumonie'}

# Correspondance pathologie -> regions anatomiques.
# Les pathologies diffuses sont restreintes aux zones moyennes, inferieures
# et hilaires (pas les apex ni "lung" entier) pour eviter un masque couvrant
# presque tout le thorax, qui ne contraindrait plus rien.
# -> avec cette correspondance : 7554 images exploitables, couverture 37.8%
ZONES_CIBLEES=['right mid lung zone','right lower lung zone',
               'left mid lung zone','left lower lung zone',
               'right hilar structures','left hilar structures']
REGIONS={
 'Pleural Effusion':['left costophrenic angle','right costophrenic angle',
                     'left lower lung zone','right lower lung zone',
                     'left hemidiaphragm','right hemidiaphragm'],
 'Edema':ZONES_CIBLEES,
 'Cardiomegaly':['left cardiac silhouette','right cardiac silhouette','cardiac silhouette'],
 'Atelectasis':ZONES_CIBLEES,
 'Pneumonia':ZONES_CIBLEES,
}

ap=argparse.ArgumentParser()
ap.add_argument('--conditions',default='Baseline,Lconsist,Lground,Lground_Lconsist')
ap.add_argument('--seeds',default='0,1,2',help='graines separees par des virgules')
ap.add_argument('--carte_dicom',default='',help='dicom_id (sans .jpg) : sauvegarde sa carte attention 14x14')
ap.add_argument('--lambda_g',type=float,default=1.0)
ap.add_argument('--lambda_c',type=float,default=2.0)
ap.add_argument('--epochs',type=int,default=25)
args=ap.parse_args()
CONDITIONS=[c.strip() for c in args.conditions.split(',') if c.strip()]
SEEDS=[int(s.strip()) for s in args.seeds.split(',') if s.strip()]
LG,LC=args.lambda_g,args.lambda_c
device='cuda' if torch.cuda.is_available() else 'cpu'
print('=== L_ground : ancrage anatomique ===', flush=True)
print(f'Conditions : {CONDITIONS} | lambda_g={LG} lambda_c={LC} | device={device}', flush=True)

# =====================================================================
# 1) Donnees du projet
# =====================================================================
pf16=os.path.join(PROC,'image_patches_f16.npy')
patches=np.load(pf16,mmap_mode='r') if os.path.exists(pf16) else np.load(os.path.join(PROC,'image_patches_clean.npy'),mmap_mode='r')
index=pd.read_csv(os.path.join(PROC,'image_index_clean.csv'))
row_of={str(d):i for i,d in enumerate(index['dicom_id'])}
ehr=pd.read_csv(os.path.join(RAW,'cohort_ehr_text_enriched.csv'))
chex=pd.read_csv(os.path.join(RAW,'labels_detection.csv'))
for m in MALADIES:
    lab=dict(zip(chex['study_id'],(chex[m]==1.0).astype(int)))
    index[m]=index['study_id'].map(lab)
index=index.dropna(subset=MALADIES).reset_index(drop=True)
for m in MALADIES: index[m]=index[m].astype(int)

# =====================================================================
# 2) Masques anatomiques a partir de Chest ImaGenome
# =====================================================================
bo_path=os.path.join(RAW,'imagenome_boxes_corrige.csv')
if not os.path.exists(bo_path):
    print('ERREUR : imagenome_boxes_corrige.csv introuvable. Lancer 31b_corriger_dimensions.py.', flush=True); sys.exit(1)
boxes=pd.read_csv(bo_path)
boxes['region']=boxes['region'].astype(str).str.strip().str.lower()
dispo=set(boxes['region'].unique())
print(f'\nBoites ImaGenome : {len(boxes)} lignes, {boxes["dicom_id"].nunique()} images', flush=True)

print('\n--- Verification de la correspondance pathologie -> regions ---', flush=True)
REG_OK={}
for mal,regs in REGIONS.items():
    presentes=[r for r in regs if r.lower() in dispo]
    absentes=[r for r in regs if r.lower() not in dispo]
    REG_OK[mal]=set(r.lower() for r in presentes)
    print(f'  {NOMS[mal]:14s}: {len(presentes)} regions trouvees', flush=True)
    if absentes: print(f'      (non presentes dans les donnees : {absentes})', flush=True)
    if not presentes:
        print(f'      ATTENTION : aucune region pour {mal}, L_ground sera inactif pour cette pathologie.', flush=True)

def boite_vers_grille(x1,y1,x2,y2,W,H):
    """Applique resize(petit cote=224) + centre-crop(224), puis normalise en 0-1."""
    if not W or not H or W<=0 or H<=0: return None
    s=224.0/min(W,H); nW,nH=W*s,H*s
    dx,dy=(nW-224.0)/2.0,(nH-224.0)/2.0
    a=[x1*s-dx, y1*s-dy, x2*s-dx, y2*s-dy]
    a=[max(0.0,min(224.0,v)) for v in a]
    if a[2]-a[0]<4 or a[3]-a[1]<4: return None
    return [a[0]/224.0,a[1]/224.0,a[2]/224.0,a[3]/224.0]

def poser(masque,b):
    c1=int(np.floor(b[0]*14)); c2=int(np.ceil(b[2]*14))
    r1=int(np.floor(b[1]*14)); r2=int(np.ceil(b[3]*14))
    masque[max(0,r1):min(14,r2), max(0,c1):min(14,c2)]=1.0

# regroupement des boites par image, en ne gardant que les regions utiles
utiles=set().union(*REG_OK.values()) if REG_OK else set()
boxes=boxes[boxes['region'].isin(utiles)]
par_image={}
for dic,g in boxes.groupby('dicom_id'):
    par_image[str(dic)]=g

# masque par (image, pathologies positives)
etiquettes={str(r['dicom_id']):[int(r[m]) for m in MALADIES] for _,r in index.iterrows()}
masques={}; sans_masque=0
for dic,g in par_image.items():
    y=etiquettes.get(dic)
    if y is None: continue
    positives=[MALADIES[k] for k in range(len(MALADIES)) if y[k]==1]
    if not positives: continue
    regs=set().union(*[REG_OK[m] for m in positives]) if positives else set()
    if not regs: continue
    m14=np.zeros((14,14),dtype=np.float32); pose=False
    for _,r in g.iterrows():
        if r['region'] not in regs: continue
        b=boite_vers_grille(float(r['x1']),float(r['y1']),float(r['x2']),float(r['y2']),
                            float(r['image_width']),float(r['image_height']))
        if b is None: continue
        poser(m14,b); pose=True
    if pose and m14.sum()>0: masques[dic]=m14.reshape(-1)
    else: sans_masque+=1

print(f'\nImages disposant d un masque anatomique exploitable : {len(masques)}', flush=True)
print(f'  (images positives sans masque utilisable : {sans_masque})', flush=True)
if len(masques)<500:
    print('ATTENTION : trop peu de masques, L_ground aura peu d effet.', flush=True)
couverture=np.mean([m.mean() for m in masques.values()]) if masques else 0
print(f'  proportion moyenne de la grille couverte par le masque : {couverture:.1%}', flush=True)

# =====================================================================
# 3) Decoupage par patient
# =====================================================================
random.seed(42); pats=list(index['subject_id'].unique()); random.shuffle(pats)
n=len(pats); a=int(.70*n); b=int(.85*n)
p_tr,p_va,p_te=set(pats[:a]),set(pats[a:b]),set(pats[b:])
def mk(ps):
    s=index[index.subject_id.isin(ps)]
    return [(row_of[str(r['dicom_id'])],r['subject_id'],str(r['dicom_id']),
             [int(r[m]) for m in MALADIES]) for _,r in s.iterrows() if str(r['dicom_id']) in row_of]
train_s,val_s,test_s=mk(p_tr),mk(p_va),mk(p_te)
n_masq_tr=sum(1 for *_,d,_ in [(x[0],x[1],x[2],x[3]) for x in train_s] if d in masques)
print(f'\ntrain {len(train_s)} / val {len(val_s)} / test {len(test_s)}', flush=True)
print(f'  dont {n_masq_tr} images d entrainement avec masque (L_ground actif)', flush=True)

# =====================================================================
# 4) Modele
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
    def forward(self,p,t,attn_grad=False):
        img=self.proj(p)
        fus,attn=self.cross(t,img,img,need_weights=attn_grad,average_attn_weights=True)
        return (self.h_fus(fus.mean(1)),self.h_img(img.mean(1)),self.h_txt(t.mean(1)),attn)

def get(i): return torch.tensor(np.asarray(patches[i]),dtype=torch.float32)
Y=np.array([y for *_,y in train_s]); pw=[]
for k in range(len(MALADIES)):
    npos=Y[:,k].sum(); pw.append(max(len(Y)-npos,1)/max(npos,1))
pw=torch.tensor(pw,dtype=torch.float32).to(device)

def coherence(img_logit,txt_logit):
    """KL symetrique entre les predictions image et texte."""
    pi=torch.sigmoid(img_logit); pt=torch.sigmoid(txt_logit); eps=1e-7
    Pi=torch.stack([pi,1-pi],dim=-1).clamp(eps,1); Pt=torch.stack([pt,1-pt],dim=-1).clamp(eps,1)
    kl=lambda x,y:(x*(x.log()-y.log())).sum(-1)
    return (kl(Pi,Pt)+kl(Pt,Pi)).mean()

def ancrage(attn,masque_t):
    """Part de l attention tombant hors du masque anatomique."""
    a=attn.mean(1).squeeze(0).clamp(min=0)          # (196,)
    dedans=(a*masque_t).sum(); total=a.sum()+1e-8
    return 1.0-dedans/total

# =====================================================================
# 5) Entrainement
# =====================================================================
def entrainer(cond,graine=0):
    use_g='Lground' in cond
    use_c='Lconsist' in cond
    torch.manual_seed(graine); np.random.seed(graine); model=Modele(DIM,len(MALADIES)).to(device)
    bce=nn.BCEWithLogitsLoss(pos_weight=pw); opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    best=1e9; bs=None; wait=0
    for ep in range(args.epochs):
        random.shuffle(train_s); model.train(); ng=0
        for (i,subj,dic,y) in train_s:
            pi=get(i).unsqueeze(0).to(device); te=EMB[subj].unsqueeze(0).to(device)
            yt=torch.tensor([y],dtype=torch.float32).to(device)
            fus,img,txt,attn=model(pi,te,attn_grad=use_g)
            loss=bce(fus,yt)+bce(img,yt)+bce(txt,yt)
            if use_c: loss=loss+LC*coherence(img,txt)
            if use_g and dic in masques:
                mt=torch.tensor(masques[dic],dtype=torch.float32,device=device)
                loss=loss+LG*ancrage(attn,mt); ng+=1
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); lv=0
        with torch.no_grad():
            for (i,subj,dic,y) in val_s:
                pi=get(i).unsqueeze(0).to(device); te=EMB[subj].unsqueeze(0).to(device)
                lv+=bce(model(pi,te)[0],torch.tensor([y],dtype=torch.float32).to(device)).item()
        lv/=max(len(val_s),1)
        if ep==0 and use_g: print(f'    (L_ground appliquee sur {ng} echantillons par epoque)', flush=True)
        if lv<best: best=lv; bs={k:v.clone() for k,v in model.state_dict().items()}; wait=0
        else:
            wait+=1
            if wait>=4: break
    model.load_state_dict(bs); model.eval(); return model

# =====================================================================
# 6) Evaluations
# =====================================================================
def seuils_optimaux(Pv,Yv):
    s=[]
    for k in range(len(MALADIES)):
        bf,bs_=0,0.5
        for t in np.arange(0.1,0.9,0.05):
            f=f1_score(Yv[:,k],(Pv[:,k]>t).astype(int),zero_division=0)
            if f>bf: bf,bs_=f,t
        s.append(bs_)
    return np.array(s)

def classification(model):
    def collecte(S):
        P,Yv=[],[]
        with torch.no_grad():
            for (i,subj,dic,y) in S:
                pi=get(i).unsqueeze(0).to(device); te=EMB[subj].unsqueeze(0).to(device)
                P.append(torch.sigmoid(model(pi,te)[0]).squeeze(0).cpu().numpy()); Yv.append(y)
        return np.array(P),np.array(Yv)
    Pv,Yv=collecte(val_s); Pt,Yt=collecte(test_s)
    s=seuils_optimaux(Pv,Yv)
    aur=[roc_auc_score(Yt[:,k],Pt[:,k]) if len(np.unique(Yt[:,k]))>1 else np.nan for k in range(len(MALADIES))]
    pred=np.stack([(Pt[:,k]>s[k]).astype(int) for k in range(len(MALADIES))],axis=1)
    return {'AUROC_moyen':float(np.nanmean(aur)),
            'F1_macro':f1_score(Yt,pred,average='macro',zero_division=0),
            'Hamming':hamming_loss(Yt,pred),
            'AUROC_par_maladie':[round(float(x),4) for x in aur]}

def carte_attention(model,i,subj):
    pi=get(i).unsqueeze(0).to(device); te=EMB[subj].unsqueeze(0).to(device)
    with torch.no_grad():
        _,_,_,attn=model(pi,te,attn_grad=True)
    return attn.mean(1).squeeze(0).cpu().numpy().reshape(14,14)

def masse_dans_masque(carte,masque_plat):
    a=np.clip(carte.reshape(-1),0,None)
    return float((a*masque_plat).sum()/(a.sum()+1e-8))

def carte_vers_boite(c,seuil=0.5):
    c=(c-c.min())/(c.max()-c.min()+1e-8)
    ys,xs=np.where(c>=seuil)
    if len(xs)==0: return None
    return [xs.min()/14.0,ys.min()/14.0,(xs.max()+1)/14.0,(ys.max()+1)/14.0]

def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    u=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/u if u>0 else 0.0

def masque_vers_boite(mp):
    m=mp.reshape(14,14); ys,xs=np.where(m>0)
    if len(xs)==0: return None
    return [xs.min()/14.0,ys.min()/14.0,(xs.max()+1)/14.0,(ys.max()+1)/14.0]

def ancrage_imagenome(model):
    """Ancrage mesure sur les patients de test, contre les regions ImaGenome."""
    masses=[]; ious=[]; hits=[]
    for (i,subj,dic,y) in test_s:
        if dic not in masques: continue
        carte=carte_attention(model,i,subj); mp=masques[dic]
        masses.append(masse_dans_masque(carte,mp))
        bp=carte_vers_boite(carte); bg=masque_vers_boite(mp)
        if bp and bg:
            ious.append(iou(bp,bg))
            k=int(np.argmax(carte)); r,c=divmod(k,14)
            hits.append(1.0 if mp.reshape(14,14)[r,c]>0 else 0.0)
    return {'n':len(masses),
            'masse_dans_region':round(float(np.mean(masses)),4) if masses else 0,
            'IoU':round(float(np.mean(ious)),4) if ious else 0,
            'pointing':round(float(np.mean(hits)),4) if hits else 0}

# ---- MS-CXR : verification independante ----
def charger_mscxr():
    cands=[os.path.join(RAW,f) for f in os.listdir(RAW) if 'ms_cxr' in f.lower() and f.endswith('.csv')]
    if not cands: return []
    ms=pd.read_csv(cands[0]); out=[]
    brut={}
    for _,r in ms.iterrows():
        dic=str(r['dicom_id'])
        if dic not in row_of: continue
        cat=str(r['category_name'])
        if cat not in MALADIES: continue
        bg=boite_vers_grille(float(r['x']),float(r['y']),float(r['x'])+float(r['w']),
                             float(r['y'])+float(r['h']),float(r['image_width']),float(r['image_height']))
        if bg is None: continue
        brut.setdefault((dic,cat,str(r.get('label_text',''))),[]).append(bg)
    for (dic,cat,_),bs_ in brut.items():
        out.append({'dicom_id':dic,'ligne':row_of[dic],'categorie':cat,
                    'boite':[min(b[0] for b in bs_),min(b[1] for b in bs_),
                             max(b[2] for b in bs_),max(b[3] for b in bs_)]})
    return out

try:
    ANNOTS=charger_mscxr()
except Exception as e:
    print('MS-CXR indisponible :',e, flush=True); ANNOTS=[]
print(f'\nConstatations MS-CXR disponibles pour verification : {len(ANNOTS)}', flush=True)
sujet_de={str(r['dicom_id']):r['subject_id'] for _,r in index.iterrows()}

def ancrage_mscxr(model):
    if not ANNOTS: return {'n':0,'IoU':0,'pointing':0}
    ious=[]; hits=[]
    for an in ANNOTS:
        carte=carte_attention(model,an['ligne'],sujet_de[an['dicom_id']])
        bp=carte_vers_boite(carte)
        if bp is None: continue
        ious.append(iou(bp,an['boite']))
        k=int(np.argmax(carte)); r,c=divmod(k,14)
        pt=((c+0.5)/14.0,(r+0.5)/14.0); b=an['boite']
        hits.append(1.0 if (b[0]<=pt[0]<=b[2] and b[1]<=pt[1]<=b[3]) else 0.0)
    return {'n':len(ious),'IoU':round(float(np.mean(ious)),4) if ious else 0,
            'pointing':round(float(np.mean(hits)),4) if hits else 0}

# =====================================================================
# 7) Boucle principale
# =====================================================================
CUMUL=os.path.join(EXP,'lground_cumul.json')
resultats=json.load(open(CUMUL)) if os.path.exists(CUMUL) else []
resultats=[r for r in resultats if r['condition'] not in CONDITIONS]
if resultats: print('\nDeja presents :',sorted({(r['condition'],r.get('graine',0)) for r in resultats}), flush=True)

deja={(r['condition'],r.get('graine',0)) for r in resultats}
for cond in CONDITIONS:
    for graine in SEEDS:
        if (cond,graine) in deja:
            print(f'\n--- {cond} | graine {graine} : deja fait, saute ---', flush=True); continue
        print(f'\n--- {cond} | graine {graine} ---', flush=True)
        model=entrainer(cond,graine)
        cl=classification(model); ag=ancrage_imagenome(model); am=ancrage_mscxr(model)
        resultats.append({'condition':cond,'graine':graine,'lambda_g':LG,'lambda_c':LC,
                          **{f'cl_{k}':v for k,v in cl.items()},
                          **{f'ig_{k}':v for k,v in ag.items()},
                          **{f'ms_{k}':v for k,v in am.items()}})
        print(f'  classification : AUROC={cl["AUROC_moyen"]:.3f} F1_macro={cl["F1_macro"]:.3f} Hamming={cl["Hamming"]:.3f}', flush=True)
        print(f'  ancrage ImaGenome (n={ag["n"]}) : masse_dans_region={ag["masse_dans_region"]:.3f} IoU={ag["IoU"]:.3f} pointing={ag["pointing"]:.3f}', flush=True)
        print(f'  ancrage MS-CXR    (n={am["n"]}) : IoU={am["IoU"]:.3f} pointing={am["pointing"]:.3f}', flush=True)
        with open(CUMUL,'w') as f: json.dump(resultats,f,indent=2)
        print(f'  -> {cond} graine {graine} sauvegardee', flush=True)

        # --- sauvegarde des POIDS du modele (pour la generation sans reentrainer) ---
        if 'Lground' in cond:
            wpath=os.path.join(EXP,f'poids_{cond}_g{graine}.pt')
            torch.save(model.state_dict(),wpath)
            print(f'  -> poids sauvegardes : {wpath}', flush=True)

        # --- sauvegarde optionnelle d une carte d attention pour la figure ---
        if args.carte_dicom and 'Lground' in cond:
            dic=args.carte_dicom.replace('.jpg','')
            sub=index[index['dicom_id'].astype(str)==dic]
            if len(sub)>0:
                ii=row_of[dic]; ss=int(sub.iloc[0]['subject_id'])
                c=carte_attention(model,ii,ss)
                c=np.clip(c,0,None); c=c/(c.max()+1e-8)
                fout=os.path.join(EXP,f'attention_{dic}.npy')
                np.save(fout,c)
                print(f'  -> carte attention sauvegardee : {fout}', flush=True)
            else:
                print(f'  !! dicom {dic} introuvable dans l index', flush=True)

# =====================================================================
# 8) Synthese et graphes
# =====================================================================
ORDRE=['Baseline','Lconsist','Lground','Lground_Lconsist']
presentes=[c for c in ORDRE if any(r['condition']==c for r in resultats)]
df=pd.DataFrame([{k:v for k,v in r.items() if k!='cl_AUROC_par_maladie'} for r in resultats])
df.to_csv(os.path.join(EXP,'lground_synthese.csv'),index=False)
print('\n=== RESULTATS BRUTS (toutes graines) ===', flush=True)
print(df.to_string(index=False), flush=True)

# ---- synthese moyenne +/- ecart-type par condition ----
METRIQUES=['cl_AUROC_moyen','cl_F1_macro','cl_Hamming',
           'ig_masse_dans_region','ig_pointing','ms_pointing']
print('\n=== SYNTHESE MOYENNE +/- ECART-TYPE (sur les graines) ===', flush=True)
synth={}
for c in ORDRE:
    sub=[r for r in resultats if r['condition']==c]
    if not sub: continue
    ligne={}
    for m in METRIQUES:
        vals=[r[m] for r in sub if m in r]
        ligne[m]=(float(np.mean(vals)),float(np.std(vals)),len(vals))
    synth[c]=ligne
    print(f'\n  {c} (n={len(sub)} graines)', flush=True)
    for m in METRIQUES:
        mo,ec,n=ligne[m]
        print(f'    {m:22s}: {mo:.3f} +/- {ec:.3f}', flush=True)

# ---- effet de L_ground vs bruit, sur la metrique reine (masse) ----
if 'Baseline' in synth and 'Lground' in synth:
    print('\n=== EFFET DE L_GROUND (masse dans la region) ===', flush=True)
    b_mo,b_ec,_=synth['Baseline']['ig_masse_dans_region']
    g_mo,g_ec,_=synth['Lground']['ig_masse_dans_region']
    diff=g_mo-b_mo; bruit=b_ec+g_ec
    print(f'  Baseline : {b_mo:.3f} +/- {b_ec:.3f}', flush=True)
    print(f'  Lground  : {g_mo:.3f} +/- {g_ec:.3f}', flush=True)
    print(f'  Difference : +{diff:.3f}  |  bruit combine : +/-{bruit:.3f}', flush=True)
    if diff>3*bruit:
        print('  VERDICT : effet MASSIF, tres largement au-dessus du bruit.', flush=True)
    elif diff>bruit:
        print('  VERDICT : effet reel (depasse le bruit).', flush=True)
    else:
        print('  VERDICT : dans le bruit, prudence.', flush=True)

print('\n=== SYNTHESE ===', flush=True)
print(df.to_string(index=False), flush=True)
if len(presentes)<len(ORDRE):
    manq=[c for c in ORDRE if c not in presentes]
    print('\nConditions manquantes :',manq, flush=True)
    print('Relancer avec : python 32_lground.py --conditions',','.join(manq), flush=True)

def val(c,cle):
    vals=[r[cle] for r in resultats if r['condition']==c and cle in r]
    return float(np.mean(vals)) if vals else 0

fig,axes=plt.subplots(1,3,figsize=(16,5))
paires=[('ig_masse_dans_region','Attention dans la region anatomique (haut=mieux)'),
        ('ig_pointing','Pointing game ImaGenome (haut=mieux)'),
        ('cl_F1_macro','F1 macro — performance de classification')]
for ax,(cle,titre) in zip(axes,paires):
    v=[val(c,cle) for c in presentes]
    ax.bar(range(len(presentes)),v,color=['#9AA7B2','#0E7C86','#B07D2B','#2E7D5B'][:len(presentes)])
    ax.set_xticks(range(len(presentes)))
    ax.set_xticklabels([c.replace('_','\n+') for c in presentes],fontsize=9)
    ax.set_title(titre,fontsize=11); ax.grid(axis='y',alpha=0.3)
    for k,x in enumerate(v): ax.text(k,x,f'{x:.3f}',ha='center',va='bottom',fontsize=9)
plt.suptitle("Contribution L_ground : ancrage anatomique et performance",fontsize=12)
plt.tight_layout(); plt.savefig(os.path.join(EXP,'lground_comparaison.png'),dpi=150,bbox_inches='tight')

fig,ax=plt.subplots(figsize=(9,5))
x=np.arange(len(presentes)); w=0.35
ax.bar(x-w/2,[val(c,'ig_IoU') for c in presentes],w,label='IoU ImaGenome',color='#0E7C86')
ax.bar(x+w/2,[val(c,'ms_IoU') for c in presentes],w,label='IoU MS-CXR (experts)',color='#B07D2B')
ax.set_xticks(x); ax.set_xticklabels([c.replace('_','\n+') for c in presentes],fontsize=9)
ax.set_ylabel('IoU'); ax.set_title("Ancrage : regions anatomiques et verification experte")
ax.legend(); ax.grid(axis='y',alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(EXP,'lground_iou.png'),dpi=150,bbox_inches='tight')

print('\n=== TERMINE ===', flush=True)
print('Fichiers dans',EXP, flush=True)
print('  lground_synthese.csv, lground_cumul.json, lground_comparaison.png, lground_iou.png', flush=True)
