#!/usr/bin/env python3
# =====================================================================
# 31_preparer_imagenome.py
# PREPARATION DES BOITES ANATOMIQUES (Chest ImaGenome).
#
# scene_graph.zip contient un fichier JSON par image (~240 000 fichiers).
# Ce script ne deplie pas l archive : il la parcourt, ne lit que les
# images presentes dans notre cohorte, et produit un CSV compact.
#
# A lancer sur le noeud de connexion (aucun GPU necessaire).
# Sortie : data/raw/imagenome_boxes.csv
#          colonnes -> dicom_id, region, x1, y1, x2, y2, image_width, image_height
# =====================================================================

import os, json, zipfile, sys
import pandas as pd

BASE=os.path.expanduser('~/scratch/MultimodalVLM')
RAW=os.path.join(BASE,'data/raw'); PROC=os.path.join(BASE,'data/processed')
ZIP=os.path.join(RAW,'scene_graph.zip')
SORTIE=os.path.join(RAW,'imagenome_boxes.csv')

if not os.path.exists(ZIP):
    print('ERREUR : archive introuvable ->', ZIP, flush=True); sys.exit(1)

index=pd.read_csv(os.path.join(PROC,'image_index_clean.csv'))
besoin=set(str(d) for d in index['dicom_id'])
print('Images de la cohorte a retrouver :', len(besoin), flush=True)

z=zipfile.ZipFile(ZIP)
membres=[n for n in z.namelist() if n.endswith('.json')]
print('Fichiers JSON dans l archive :', len(membres), flush=True)

# ---------------------------------------------------------------
# 1) Inspection de la structure sur un premier fichier
# ---------------------------------------------------------------
apercu=json.loads(z.read(membres[0]))
print('\n--- Structure d un graphe de scene ---', flush=True)
print('Cles de premier niveau :', list(apercu.keys()), flush=True)
cle_objets=None
for c in ['objects','object','bboxes','regions']:
    if c in apercu and isinstance(apercu[c],list) and apercu[c]:
        cle_objets=c; break
if cle_objets is None:
    print('ERREUR : liste des objets introuvable. Cles disponibles ci-dessus.', flush=True); sys.exit(1)
print(f'Liste des regions trouvee sous la cle : "{cle_objets}"', flush=True)
print('Cles d un objet :', list(apercu[cle_objets][0].keys()), flush=True)

# ---------------------------------------------------------------
# 2) Detection automatique des champs
# ---------------------------------------------------------------
exemple=apercu[cle_objets][0]
def premier_present(dico,*noms):
    for n in noms:
        if n in dico: return n
    return None

f_nom=premier_present(exemple,'bbox_name','name','object_name','region')
f_x1=premier_present(exemple,'original_x1','x1','bbox_x1')
f_y1=premier_present(exemple,'original_y1','y1','bbox_y1')
f_x2=premier_present(exemple,'original_x2','x2','bbox_x2')
f_y2=premier_present(exemple,'original_y2','y2','bbox_y2')
f_w =premier_present(exemple,'width','original_width','image_width')
f_h =premier_present(exemple,'height','original_height','image_height')
print(f'Champs retenus -> nom:{f_nom}  boite:({f_x1},{f_y1},{f_x2},{f_y2})  taille:({f_w},{f_h})', flush=True)
if not all([f_nom,f_x1,f_y1,f_x2,f_y2]):
    print('ERREUR : champs de boite introuvables. Voir les cles ci-dessus.', flush=True); sys.exit(1)

# ---------------------------------------------------------------
# 3) Extraction, limitee aux images de la cohorte
# ---------------------------------------------------------------
def id_depuis_chemin(chemin):
    base=os.path.basename(chemin)
    for suffixe in ['_SceneGraph.json','_scenegraph.json','.json']:
        if base.endswith(suffixe): return base[:-len(suffixe)]
    return base

lignes=[]; trouvees=set(); traites=0
for n in membres:
    dic=id_depuis_chemin(n)
    if dic not in besoin: continue
    try:
        d=json.loads(z.read(n))
    except Exception:
        continue
    trouvees.add(dic); traites+=1
    W=d.get(f_w) if f_w else None; H=d.get(f_h) if f_h else None
    for ob in d.get(cle_objets,[]):
        try:
            x1=float(ob[f_x1]); y1=float(ob[f_y1]); x2=float(ob[f_x2]); y2=float(ob[f_y2])
        except (KeyError,TypeError,ValueError):
            continue
        w=ob.get(f_w,W) if f_w else W
        h=ob.get(f_h,H) if f_h else H
        lignes.append({'dicom_id':dic,'region':ob.get(f_nom,''),
                       'x1':x1,'y1':y1,'x2':x2,'y2':y2,
                       'image_width':w,'image_height':h})
    if traites % 2000 == 0:
        print(f'  ... {traites} images de la cohorte traitees, {len(lignes)} regions extraites', flush=True)

df=pd.DataFrame(lignes)
df.to_csv(SORTIE,index=False)

print('\n=== TERMINE ===', flush=True)
print('Images de la cohorte retrouvees dans ImaGenome :', len(trouvees), f'/ {len(besoin)}', flush=True)
print('Regions extraites au total :', len(df), flush=True)
if len(df):
    print('Regions par image (moyenne) :', round(len(df)/max(len(trouvees),1),1), flush=True)
    print('\n--- Noms de regions disponibles ---', flush=True)
    for nom,cnt in df['region'].value_counts().items():
        print(f'  {nom:38s} {cnt}', flush=True)
manque_taille = df[['image_width','image_height']].isna().any(axis=1).sum() if len(df) else 0
if manque_taille:
    print(f'\nAttention : {manque_taille} lignes sans dimensions d image (a completer depuis les JPG).', flush=True)
print('\nFichier ecrit :', SORTIE, flush=True)
