#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys, json, math, time, shutil, threading, subprocess, importlib.util
from pathlib import Path
from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
try:
    from PIL import Image, ImageTk
    Image.MAX_IMAGE_PIXELS = None
except Exception as e:
    raise SystemExit('Install Pillow: python -m pip install pillow\n'+str(e))
try:
    import numpy as np
except Exception as e:
    raise SystemExit('Install numpy: python -m pip install numpy\n'+str(e))
APP='Mustatil All GUIs + Crop Annotator + SAM2 Preview + FormLearner Preview'
IMG_EXT={'.jpg','.jpeg','.png','.tif','.tiff','.bmp','.webp'}
CLASSES=['mustatil','false_positive']

def have(m): return importlib.util.find_spec(m) is not None

def deps():
    return '\n'.join([f'Python: {sys.executable}']+[f'{m}: {"OK" if have(m) else "missing/optional"}' for m in ['ultralytics','torch','PIL','numpy','cv2','yaml','rasterio','onnxruntime']])

def is_tif(p): return Path(p).suffix.lower() in {'.tif','.tiff'}

def open_img(path, log):
    p=Path(path)
    if is_tif(p):
        if not have('rasterio'): raise RuntimeError('TIFF/GeoTIFF braucht rasterio: python -m pip install rasterio')
        import rasterio
        r=rasterio.open(p); log(f'rasterio streaming: {r.width}x{r.height}, crs={r.crs}')
        return r.width,r.height,'rasterio',r
    im=Image.open(p); log(f'PIL image: {im.width}x{im.height}'); return im.width,im.height,'pil',im

def read_tile(reader,mode,x,y,size,W,H):
    w=min(size,W-x); h=min(size,H-y)
    if mode=='rasterio':
        from rasterio.windows import Window
        arr=reader.read([1,2,3] if reader.count>=3 else [1], window=Window(x,y,w,h), boundless=True, fill_value=0)
        if arr.shape[0]==1: arr=np.repeat(arr,3,axis=0)
        arr=np.moveaxis(arr,0,-1)
        if arr.dtype!=np.uint8:
            arr=arr.astype('float32'); mn=float(np.nanmin(arr)); mx=float(np.nanmax(arr))
            if mx>mn: arr=(arr-mn)/(mx-mn)*255
            arr=np.clip(arr,0,255).astype('uint8')
        return Image.fromarray(arr,'RGB')
    return reader.crop((x,y,x+w,y+h)).convert('RGB')

def load_preview(path,maxs=2400):
    p=Path(path)
    if is_tif(p):
        if not have('rasterio'): raise RuntimeError('TIFF Preview braucht rasterio')
        import rasterio
        with rasterio.open(p) as r:
            s=min(maxs/max(1,r.width),maxs/max(1,r.height),1); ow=max(1,int(r.width*s)); oh=max(1,int(r.height*s))
            arr=r.read([1,2,3] if r.count>=3 else [1], out_shape=(3 if r.count>=3 else 1,oh,ow))
            if arr.shape[0]==1: arr=np.repeat(arr,3,axis=0)
            arr=np.moveaxis(arr,0,-1)
            if arr.dtype!=np.uint8:
                arr=arr.astype('float32'); mn=float(np.nanmin(arr)); mx=float(np.nanmax(arr))
                if mx>mn: arr=(arr-mn)/(mx-mn)*255
                arr=np.clip(arr,0,255).astype('uint8')
            return Image.fromarray(arr,'RGB'),r.width,r.height
    im=Image.open(p).convert('RGB'); W,H=im.size; im.thumbnail((maxs,maxs)); return im.copy(),W,H

def positions(W,H,tile,overlap,shifted=True):
    stride=max(1,tile-overlap); out=[]
    for off in ([0,stride//2] if shifted and stride//2>0 else [0]):
        y=off
        while y<H:
            x=off
            while x<W:
                out.append((x,y));
                if x+tile>=W: break
                x+=stride
            if y+tile>=H: break
            y+=stride
    seen=set(); res=[]
    for p in out:
        if p not in seen: seen.add(p); res.append(p)
    return res

def iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1=max(ax1,bx1); iy1=max(ay1,by1); ix2=min(ax2,bx2); iy2=min(ay2,by2)
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    if inter<=0: return 0
    aa=max(0,ax2-ax1)*max(0,ay2-ay1); bb=max(0,bx2-bx1)*max(0,by2-by1)
    return inter/(aa+bb-inter+1e-9)

def write_label(path,bbox,left,top,w,h,cls):
    x1,y1,x2,y2=bbox; lx1=max(0,min(w,x1-left)); lx2=max(0,min(w,x2-left)); ly1=max(0,min(h,y1-top)); ly2=max(0,min(h,y2-top))
    if lx2<=lx1 or ly2<=ly1: Path(path).write_text('',encoding='utf-8'); return
    cx=((lx1+lx2)/2)/w; cy=((ly1+ly2)/2)/h; bw=(lx2-lx1)/w; bh=(ly2-ly1)/h
    Path(path).write_text(f'{cls} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}\n',encoding='utf-8')

def read_boxes(img,label):
    if not Path(label).exists(): return []
    W,H=Image.open(img).size; out=[]
    for line in Path(label).read_text(encoding='utf-8',errors='ignore').splitlines():
        p=line.split()
        if len(p)!=5: continue
        c=int(float(p[0])); cx,cy,bw,bh=map(float,p[1:])
        out.append([c,(cx-bw/2)*W,(cy-bh/2)*H,(cx+bw/2)*W,(cy+bh/2)*H])
    return out

def save_boxes(img,label,boxes):
    W,H=Image.open(img).size; lines=[]
    for c,x1,y1,x2,y2 in boxes:
        x1,x2=sorted((max(0,x1),min(W,x2))); y1,y2=sorted((max(0,y1),min(H,y2)))
        if x2>x1 and y2>y1:
            lines.append(f'{int(c)} {((x1+x2)/2/W):.8f} {((y1+y2)/2/H):.8f} {((x2-x1)/W):.8f} {((y2-y1)/H):.8f}')
    Path(label).write_text('\n'.join(lines)+('\n' if lines else ''),encoding='utf-8')

def run_live(cmd,log,cwd=None):
    log('$ '+' '.join(map(str,cmd))); env=os.environ.copy(); env['PYTHONUTF8']='1'; env['PYTHONIOENCODING']='utf-8'
    p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',cwd=str(cwd) if cwd else None,env=env)
    for line in p.stdout: log(line.rstrip())
    if p.wait()!=0: raise RuntimeError('Command failed')

def crop_features(pil_img, bbox=None):
    import numpy as _np, math as _math
    if bbox is not None:
        x1,y1,x2,y2=map(int,map(round,bbox)); x1=max(0,x1); y1=max(0,y1); x2=min(pil_img.width,x2); y2=min(pil_img.height,y2)
        if x2>x1 and y2>y1: pil_img=pil_img.crop((x1,y1,x2,y2))
    im=pil_img.convert('RGB').resize((256,256))
    arr=_np.asarray(im).astype('float32')/255.0
    r,g,b=arr[:,:,0],arr[:,:,1],arr[:,:,2]
    green=((g>r+0.03)&(g>b+0.03)&(g>0.12)).mean()
    gray=(0.299*r+0.587*g+0.114*b)
    gx=_np.abs(_np.diff(gray,axis=1)).mean(); gy=_np.abs(_np.diff(gray,axis=0)).mean(); edge=float(gx+gy)
    mean=float(gray.mean()); std=float(gray.std())
    w,h=pil_img.size; aspect=max(w,h)/max(1,min(w,h)); area=(w*h)/(256.0*256.0)
    return [float(_math.log1p(aspect)), float(area), float(mean), float(std), float(edge), float(green)]

class SimpleFormLearner:
    def __init__(self,w=None,b=0.0,mean=None,std=None): self.w=w; self.b=b; self.mean=mean; self.std=std
    def fit(self,X,y,epochs=1200,lr=0.08,l2=0.001):
        import numpy as _np
        X=_np.asarray(X,dtype='float64'); y=_np.asarray(y,dtype='float64')
        self.mean=X.mean(axis=0); self.std=X.std(axis=0)+1e-6
        Xn=(X-self.mean)/self.std; w=_np.zeros(Xn.shape[1]); b=0.0
        for _ in range(int(epochs)):
            z=Xn@w+b; p=1/(1+_np.exp(-_np.clip(z,-40,40)))
            w-=lr*((Xn.T@(p-y))/len(y)+l2*w); b-=lr*float((p-y).mean())
        self.w=w; self.b=float(b)
    def predict(self,x):
        import numpy as _np, math as _math
        x=_np.asarray(x,dtype='float64'); x=(x-self.mean)/self.std; z=float(x@self.w+self.b)
        if z<-40: return 0.0
        if z>40: return 1.0
        return 1/(1+_math.exp(-z))
    def save(self,path):
        Path(path).write_text(json.dumps({'w':self.w.tolist(),'b':self.b,'mean':self.mean.tolist(),'std':self.std.tolist(),'features':['log_aspect','rel_area','gray_mean','gray_std','edge_density','green_ratio']},indent=2),encoding='utf-8')
    @staticmethod
    def load(path):
        import numpy as _np
        d=json.loads(Path(path).read_text(encoding='utf-8'))
        return SimpleFormLearner(_np.asarray(d['w']),float(d['b']),_np.asarray(d['mean']),_np.asarray(d['std']))

@dataclass
class Det:
    slot:int; name:str; cls:int; conf:float; x1:float; y1:float; x2:float; y2:float; score:float=0; consensus:int=1
    def bbox(self): return (self.x1,self.y1,self.x2,self.y2)


# ---------------- QGIS-safe GeoPackage fallback without Fiona/GeoPandas/GDAL ----------------
def _gpkg_epsg_int(crs_name=None):
    try:
        if crs_name is None: return -1
        t=str(crs_name).upper().strip()
        if not t or t in {'NONE','PIXEL'}: return -1
        if 'EPSG:' in t:
            digits=''.join(ch for ch in t.split('EPSG:')[-1] if ch.isdigit())
            return int(digits) if digits else -1
        if t.isdigit(): return int(t)
    except Exception:
        pass
    return -1

def _gpkg_polygon_wkb(coords):
    import struct
    rings=[]
    for ring in coords or []:
        pts=[]
        for p in ring:
            if len(p)>=2: pts.append((float(p[0]),float(p[1])))
        if len(pts)>=3:
            if pts[0]!=pts[-1]: pts.append(pts[0])
            rings.append(pts)
    if not rings: return b''
    b=bytearray(); b += struct.pack('<B',1); b += struct.pack('<I',3); b += struct.pack('<I',len(rings))
    for pts in rings:
        b += struct.pack('<I',len(pts))
        for x,y in pts: b += struct.pack('<dd',x,y)
    return bytes(b)

def _gpkg_blob(wkb,srs_id):
    import struct
    return b'GP' + struct.pack('<BBi',0,1,int(srs_id)) + wkb

def write_gpkg_fallback(path, features, layer='detections', crs_name=None, log_fn=None):
    import sqlite3, tempfile, time, shutil, json
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not features: raise RuntimeError('Keine Features zum Schreiben.')
    srs=_gpkg_epsg_int(crs_name)
    tmp=Path(tempfile.gettempdir())/f'mustatil_no_gdal_{path.stem}_{int(time.time())}.gpkg'
    if tmp.exists(): tmp.unlink()
    con=sqlite3.connect(str(tmp)); cur=con.cursor()
    try:
        cur.execute('PRAGMA application_id = 1196437808')
        cur.execute('PRAGMA user_version = 10400')
        cur.execute('CREATE TABLE gpkg_spatial_ref_sys (srs_name TEXT NOT NULL, srs_id INTEGER NOT NULL PRIMARY KEY, organization TEXT NOT NULL, organization_coordsys_id INTEGER NOT NULL, definition TEXT NOT NULL, description TEXT)')
        rows=[('Undefined Cartesian SRS',-1,'NONE',-1,'undefined','undefined cartesian coordinate reference system'),('Undefined Geographic SRS',0,'NONE',0,'undefined','undefined geographic coordinate reference system'),('WGS 84 geodetic',4326,'EPSG',4326,'EPSG:4326','longitude/latitude coordinates in decimal degrees on WGS84'),('WGS 84 / Pseudo-Mercator',3857,'EPSG',3857,'EPSG:3857','Web Mercator meters')]
        cur.executemany('INSERT INTO gpkg_spatial_ref_sys VALUES (?,?,?,?,?,?)', rows)
        if srs not in (-1,0,4326,3857): cur.execute('INSERT OR IGNORE INTO gpkg_spatial_ref_sys VALUES (?,?,?,?,?,?)',(f'EPSG:{srs}',srs,'EPSG',srs,f'EPSG:{srs}',f'EPSG:{srs}'))
        cur.execute("CREATE TABLE gpkg_contents (table_name TEXT NOT NULL PRIMARY KEY, data_type TEXT NOT NULL, identifier TEXT UNIQUE, description TEXT DEFAULT '', last_change DATETIME NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')), min_x DOUBLE, min_y DOUBLE, max_x DOUBLE, max_y DOUBLE, srs_id INTEGER)")
        cur.execute('CREATE TABLE gpkg_geometry_columns (table_name TEXT NOT NULL, column_name TEXT NOT NULL, geometry_type_name TEXT NOT NULL, srs_id INTEGER NOT NULL, z TINYINT NOT NULL, m TINYINT NOT NULL, PRIMARY KEY (table_name, column_name))')
        table='detections' if not layer else ''.join(ch if ch.isalnum() or ch=='_' else '_' for ch in layer)
        cur.execute(f'CREATE TABLE {table} (fid INTEGER PRIMARY KEY AUTOINCREMENT, geom BLOB NOT NULL, rank INTEGER, model TEXT, model_index INTEGER, confidence DOUBLE, score DOUBLE, form_score DOUBLE, consensus INTEGER, pixel_bbox TEXT, properties_json TEXT)')
        cur.execute('INSERT INTO gpkg_geometry_columns VALUES (?, ?, ?, ?, 0, 0)', (table,'geom','POLYGON',srs))
        minx=miny=float('inf'); maxx=maxy=float('-inf'); count=0
        for i,feat in enumerate(features,1):
            geom=feat.get('geometry') or {}; coords=geom.get('coordinates') or []
            wkb=_gpkg_polygon_wkb(coords)
            if not wkb: continue
            for ring in coords:
                for p in ring:
                    if len(p)>=2:
                        x=float(p[0]); y=float(p[1]); minx=min(minx,x); miny=min(miny,y); maxx=max(maxx,x); maxy=max(maxy,y)
            props=feat.get('properties') or {}
            cur.execute(f'INSERT INTO {table} (geom,rank,model,model_index,confidence,score,form_score,consensus,pixel_bbox,properties_json) VALUES (?,?,?,?,?,?,?,?,?,?)',(
                sqlite3.Binary(_gpkg_blob(wkb,srs)), int(props.get('rank') or i), str(props.get('model') or props.get('model_name') or ''), int(props.get('model_index') or 0), float(props.get('confidence') or 0.0), float(props.get('score') or props.get('ensemble_score') or 0.0), float(props.get('form_score') or 0.0), int(props.get('consensus') or props.get('consensus_count') or 0), json.dumps(props.get('pixel_bbox') or []), json.dumps(props,ensure_ascii=False)))
            count+=1
        if count==0: raise RuntimeError('Keine gültigen Polygon-Features für GeoPackage.')
        now=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        cur.execute("INSERT INTO gpkg_contents (table_name,data_type,identifier,description,last_change,min_x,min_y,max_x,max_y,srs_id) VALUES (?, 'features', ?, 'Mustatil detections', ?, ?, ?, ?, ?, ?)",(table,table,now,minx,miny,maxx,maxy,srs))
        con.commit()
    finally:
        con.close()
    final=path
    if path.exists():
        try: path.unlink()
        except Exception: final=path.with_name(path.stem+f'_new_{int(time.time())}'+path.suffix)
    shutil.copy2(tmp, final)
    try: tmp.unlink()
    except Exception: pass
    if log_fn: log_fn(f'GeoPackage ohne Fiona/GDAL geschrieben: {final}')
    return final

class GUI(tk.Tk):
    def __init__(self):
        super().__init__(); self.title(APP); self.geometry('1550x940')
        self.models=[tk.StringVar(),tk.StringVar(),tk.StringVar()]; self.image=tk.StringVar(); self.output=tk.StringVar(); self.project=tk.StringVar()
        self.conf=tk.DoubleVar(value=.05); self.showconf=tk.DoubleVar(value=.1); self.minscore=tk.DoubleVar(value=0); self.tile=tk.IntVar(value=1024); self.overlap=tk.IntVar(value=384); self.shift=tk.BooleanVar(value=True)
        self.cropdir=tk.StringVar(); self.cropsize=tk.IntVar(value=1024); self.pad=tk.IntVar(value=128); self.startcls=tk.IntVar(value=0); self.sammodel=tk.StringVar(value='sam2_b.pt')
        self.sam_source_dir=tk.StringVar(); self.sam_images=[]; self.sam_selected_img=None; self.sam_preview_img=None; self.sam_current_polys=[]
        self.sam_padding=tk.IntVar(value=96); self.sam_max_crop=tk.IntVar(value=1024); self.sam_use_ann_boxes=tk.BooleanVar(value=True); self.sam_skip_existing=tk.BooleanVar(value=True); self.sam_prompt_boxes=[]
        self.trainmodel=tk.StringVar(value='yolov8n.pt'); self.epochs=tk.IntVar(value=80); self.imgsz=tk.IntVar(value=640); self.batch=tk.IntVar(value=2); self.device=tk.StringVar(value='cpu')
        self.low_ram_mode=tk.BooleanVar(value=True); self.auto_resume=tk.BooleanVar(value=True)
        self.train_chunk_enabled=tk.BooleanVar(value=True); self.train_chunk_size=tk.IntVar(value=1024); self.train_chunk_overlap=tk.IntVar(value=128); self.train_chunk_min_visible=tk.DoubleVar(value=0.35); self.train_keep_negative_chunks=tk.BooleanVar(value=True)
        self.dets=[]; self.preview=None; self.zoom=1; self.pan=[0,0]; self.ann_imgs=[]; self.ann_i=0; self.ann_boxes=[]; self.ann_cls=tk.IntVar(value=0); self.drag=None
        self.build(); self.after(200,lambda:self.log(deps()))
    def build(self):
        nb=ttk.Notebook(self); nb.pack(fill=tk.BOTH,expand=True)
        t1=ttk.Frame(nb); t2=ttk.Frame(nb); t3=ttk.Frame(nb); t4=ttk.Frame(nb); t5=ttk.Frame(nb); t6=ttk.Frame(nb)
        nb.add(t1,text='1 Detection + Crops'); nb.add(t2,text='2 Crop-Annotator + SAM2'); nb.add(t3,text='3 YOLO Trainer')
        nb.add(t4,text='4 SAM2 Bildliste'); nb.add(t5,text='5 FormTrainer'); nb.add(t6,text='6 Detection mit FormLearner')

        # --- Tab 1: Detection + Crops ---
        left=ttk.Frame(t1,padding=8); left.pack(side=tk.LEFT,fill=tk.Y); right=ttk.Frame(t1,padding=8); right.pack(fill=tk.BOTH,expand=True)
        mf=ttk.LabelFrame(left,text='Bis zu 3 YOLO Modelle'); mf.pack(fill=tk.X)
        for i,v in enumerate(self.models):
            ttk.Label(mf,text=f'Modell {i+1}').grid(row=i,column=0); ttk.Entry(mf,textvariable=v,width=42).grid(row=i,column=1); ttk.Button(mf,text='...',command=lambda vv=v:self.pick(vv,[('YOLO','*.pt *.onnx'),('All','*.*')])).grid(row=i,column=2)
        gf=ttk.LabelFrame(left,text='Bild und Filter'); gf.pack(fill=tk.X,pady=4)
        ttk.Label(gf,text='Großbild').grid(row=0,column=0); ttk.Entry(gf,textvariable=self.image,width=42).grid(row=0,column=1); ttk.Button(gf,text='...',command=lambda:self.pick(self.image,[('Images','*.tif *.tiff *.jpg *.jpeg *.png *.bmp *.webp'),('All','*.*')])).grid(row=0,column=2)
        ttk.Button(gf,text='Preview laden',command=self.loadprev).grid(row=1,column=0,columnspan=3,sticky='ew')
        # Rechen-Confidence ist ein eigenes Eingabefeld und wird nur beim YOLO-Rechnen benutzt.
        # Anzeige-Confidence ist nur ein Vorschau-Filter und ändert keine gespeicherten/berechneten Ergebnisse.
        ttk.Label(gf,text='Rechen confidence YOLO').grid(row=2,column=0,sticky='w')
        ttk.Entry(gf,textvariable=self.conf,width=8).grid(row=2,column=1,sticky='w')
        ttk.Label(gf,text='nur für neue Detection').grid(row=2,column=2,sticky='w')
        ttk.Label(gf,text='Anzeige confidence').grid(row=3,column=0,sticky='w')
        ttk.Scale(gf,variable=self.showconf,from_=.01,to=.9,orient=tk.HORIZONTAL,command=lambda e:self.redraw()).grid(row=3,column=1,sticky='ew')
        ttk.Label(gf,textvariable=self.showconf,width=6).grid(row=3,column=2)
        ttk.Label(gf,text='Mindestscore').grid(row=4,column=0,sticky='w')
        ttk.Scale(gf,variable=self.minscore,from_=0,to=5,orient=tk.HORIZONTAL,command=lambda e:self.redraw()).grid(row=4,column=1,sticky='ew')
        ttk.Label(gf,textvariable=self.minscore,width=6).grid(row=4,column=2)
        ttk.Label(gf,text='Tile').grid(row=5,column=0); ttk.Entry(gf,textvariable=self.tile,width=8).grid(row=5,column=1,sticky='w'); ttk.Label(gf,text='Overlap').grid(row=5,column=1); ttk.Entry(gf,textvariable=self.overlap,width=8).grid(row=5,column=2)
        ttk.Checkbutton(gf,text='verschobene Tiles',variable=self.shift).grid(row=6,column=0,columnspan=3,sticky='w')
        ttk.Button(gf,text='TILED DETECTION STARTEN',command=lambda:threading.Thread(target=self.detect,daemon=True).start()).grid(row=7,column=0,columnspan=3,sticky='ew')
        cf=ttk.LabelFrame(left,text='Detectionen als 1024 PNGs speichern'); cf.pack(fill=tk.X,pady=4)
        ttk.Entry(cf,textvariable=self.cropdir,width=42).grid(row=0,column=0,columnspan=2); ttk.Button(cf,text='...',command=self.pickcrop).grid(row=0,column=2)
        ttk.Label(cf,text='Größe').grid(row=1,column=0); ttk.Entry(cf,textvariable=self.cropsize,width=8).grid(row=1,column=1,sticky='w'); ttk.Label(cf,text='Padding').grid(row=1,column=1); ttk.Entry(cf,textvariable=self.pad,width=8).grid(row=1,column=2)
        ttk.Label(cf,text='Export neutral: keine Labels werden geschrieben. Bewertung erfolgt im Crop-Annotator.').grid(row=2,column=0,columnspan=3,sticky='w')
        ttk.Button(cf,text='berechnete Detectionen als neutrale PNG-Crops exportieren',command=self.export_crops).grid(row=3,column=0,columnspan=3,sticky='ew')
        ttk.Label(left,text='Export').pack(anchor='w'); ttk.Entry(left,textvariable=self.output,width=48).pack(fill=tk.X); ttk.Button(left,text='Export GeoJSON/GPKG',command=self.export_geo).pack(fill=tk.X)
        self.logbox=tk.Text(left,width=62,height=22); self.logbox.pack(fill=tk.BOTH,expand=True)
        self.can=tk.Canvas(right,bg='black'); self.can.pack(fill=tk.BOTH,expand=True); self.can.bind('<MouseWheel>',self.wheel); self.can.bind('<ButtonPress-1>',self.press); self.can.bind('<B1-Motion>',self.move)

        # --- Tab 2: Crop-Annotator ---
        top=ttk.Frame(t2,padding=6); top.pack(fill=tk.X)
        ttk.Entry(top,textvariable=self.project,width=70).pack(side=tk.LEFT,fill=tk.X,expand=True)
        ttk.Button(top,text='Projekt...',command=self.pickproj).pack(side=tk.LEFT)
        ttk.Button(top,text='Laden',command=self.loadproj).pack(side=tk.LEFT)
        ttk.Button(top,text='<',command=lambda:self.nextann(-1)).pack(side=tk.LEFT)
        ttk.Button(top,text='>',command=lambda:self.nextann(1)).pack(side=tk.LEFT)
        ttk.Button(top,text='Bild = POSITIV',command=lambda:self.set_image_class(0)).pack(side=tk.LEFT,padx=2)
        ttk.Button(top,text='Bild = FALSE',command=lambda:self.set_image_class(1)).pack(side=tk.LEFT,padx=2)
        ttk.Button(top,text='Speichern',command=self.saveann).pack(side=tk.LEFT)

        mid=ttk.Frame(t2); mid.pack(fill=tk.BOTH,expand=True)
        ann_left=ttk.Frame(mid,padding=6); ann_left.pack(side=tk.LEFT,fill=tk.Y)
        ann_right=ttk.Frame(mid); ann_right.pack(side=tk.LEFT,fill=tk.BOTH,expand=True)
        ttk.Label(ann_left,text='Aktuelle Klasse für neue/markierte Box').pack(anchor='w')
        ttk.Radiobutton(ann_left,text='0 positiv / mustatil',variable=self.ann_cls,value=0).pack(anchor='w')
        ttk.Radiobutton(ann_left,text='1 false_positive',variable=self.ann_cls,value=1).pack(anchor='w')
        ttk.Separator(ann_left).pack(fill=tk.X,pady=6)
        ttk.Label(ann_left,text='Annotationen dieses Bildes').pack(anchor='w')
        self.ann_list=tk.Listbox(ann_left,width=42,height=18,exportselection=False)
        self.ann_list.pack(fill=tk.BOTH,expand=True)
        self.ann_list.bind('<<ListboxSelect>>',self.on_ann_select)
        ttk.Button(ann_left,text='markierte Box = gewählte Klasse',command=self.set_selected_cls).pack(fill=tk.X,pady=2)
        ttk.Button(ann_left,text='markierte Box löschen',command=self.delete_selected_box).pack(fill=tk.X,pady=2)
        ttk.Button(ann_left,text='alle Boxen löschen / neutral',command=self.clear_current_ann).pack(fill=tk.X,pady=2)
        ttk.Button(ann_left,text='aus Manifest-Detection Box neu setzen',command=lambda:self.set_image_class(int(self.ann_cls.get()))).pack(fill=tk.X,pady=2)
        ttk.Label(ann_left,text='Links ziehen = Box erstellen.\nMausrad = Zoom.\nMittlere/rechte Maustaste ziehen = Pan.\nNeutral bedeutet: keine Labeldatei/keine Box.').pack(anchor='w',pady=8)
        sam_box=ttk.LabelFrame(ann_left,text='SAM2 für aktuelles Crop / Projekt',padding=5); sam_box.pack(fill=tk.X,pady=6)
        ttk.Label(sam_box,text='SAM2 Modell').grid(row=0,column=0,sticky='w')
        ttk.Entry(sam_box,textvariable=self.sammodel,width=28).grid(row=0,column=1,sticky='ew')
        ttk.Button(sam_box,text='...',command=lambda:self.pick(self.sammodel,[('PT','*.pt'),('All','*.*')])).grid(row=0,column=2)
        ttk.Label(sam_box,text='Padding').grid(row=1,column=0,sticky='w')
        ttk.Entry(sam_box,textvariable=self.sam_padding,width=8).grid(row=1,column=1,sticky='w')
        ttk.Label(sam_box,text='MaxCrop').grid(row=1,column=1,sticky='e')
        ttk.Entry(sam_box,textvariable=self.sam_max_crop,width=8).grid(row=1,column=2,sticky='w')
        ttk.Checkbutton(sam_box,text='annotierte Kästchen als Prompts nutzen',variable=self.sam_use_ann_boxes).grid(row=2,column=0,columnspan=3,sticky='w')
        ttk.Button(sam_box,text='SAM2: aktuelles Crop segmentieren',command=lambda:threading.Thread(target=self.sam2_current_annotation,daemon=True).start()).grid(row=3,column=0,columnspan=3,sticky='ew',pady=2)
        ttk.Button(sam_box,text='SAM2: alle Crops im Projekt segmentieren',command=lambda:threading.Thread(target=self.sam2_all_annotation_project,daemon=True).start()).grid(row=4,column=0,columnspan=3,sticky='ew',pady=2)
        sam_box.columnconfigure(1,weight=1)
        self.ann_status=tk.StringVar(value='Keine Annotation geladen')
        ttk.Label(ann_left,textvariable=self.ann_status,wraplength=280).pack(anchor='w',pady=4)
        self.acan=tk.Canvas(ann_right,bg='black')
        self.acan.pack(fill=tk.BOTH,expand=True)
        self.acan.bind('<MouseWheel>',self.awheel)
        self.acan.bind('<ButtonPress-1>',self.apress)
        self.acan.bind('<ButtonRelease-1>',self.arelease)
        self.acan.bind('<ButtonPress-2>',self.apan_press); self.acan.bind('<B2-Motion>',self.apan_move)
        self.acan.bind('<ButtonPress-3>',self.apan_press); self.acan.bind('<B3-Motion>',self.apan_move)
        self.ann_zoom=1.0; self.ann_pan=[0,0]; self.ann_selected=-1; self.ann_manifest={}

        # --- Tab 3: YOLO Trainer ---
        tr=ttk.Frame(t3,padding=10); tr.pack(fill=tk.BOTH,expand=True); row=0
        ttk.Label(tr,text='Base model').grid(row=row,column=0,sticky='w',pady=4)
        ttk.Entry(tr,textvariable=self.trainmodel).grid(row=row,column=1,sticky='ew',pady=4); row+=1
        for lab,var in [('Epochs',self.epochs),('Image size',self.imgsz),('Batch',self.batch)]:
            ttk.Label(tr,text=lab).grid(row=row,column=0,sticky='w',pady=4)
            ttk.Entry(tr,textvariable=var).grid(row=row,column=1,sticky='ew',pady=4); row+=1
        ttk.Label(tr,text='Device').grid(row=row,column=0,sticky='w',pady=4)
        ttk.Combobox(tr,textvariable=self.device,values=['cpu','directml','cuda','0'],state='normal').grid(row=row,column=1,sticky='ew',pady=4); row+=1
        ttk.Checkbutton(tr,text='Low-RAM stabil mode',variable=self.low_ram_mode).grid(row=row,column=0,columnspan=2,sticky='w',pady=4); row+=1
        chunk_frame=ttk.LabelFrame(tr,text='Training image chunking for huge maps',padding=6)
        chunk_frame.grid(row=row,column=0,columnspan=2,sticky='ew',pady=6); chunk_frame.columnconfigure(1,weight=1)
        ttk.Checkbutton(chunk_frame,text='Cut project images into chunks while preparing YOLO dataset',variable=self.train_chunk_enabled).grid(row=0,column=0,columnspan=3,sticky='w',pady=2)
        ttk.Label(chunk_frame,text='Chunk size px').grid(row=1,column=0,sticky='w',pady=2)
        ttk.Entry(chunk_frame,textvariable=self.train_chunk_size,width=10).grid(row=1,column=1,sticky='w',pady=2)
        ttk.Label(chunk_frame,text='usually 1024 or 1280',foreground='#555').grid(row=1,column=2,sticky='w',padx=6)
        ttk.Label(chunk_frame,text='Overlap px').grid(row=2,column=0,sticky='w',pady=2)
        ttk.Entry(chunk_frame,textvariable=self.train_chunk_overlap,width=10).grid(row=2,column=1,sticky='w',pady=2)
        ttk.Label(chunk_frame,text='128-256 helps objects on borders',foreground='#555').grid(row=2,column=2,sticky='w',padx=6)
        ttk.Label(chunk_frame,text='Min visible label fraction').grid(row=3,column=0,sticky='w',pady=2)
        ttk.Entry(chunk_frame,textvariable=self.train_chunk_min_visible,width=10).grid(row=3,column=1,sticky='w',pady=2)
        ttk.Label(chunk_frame,text='0.35 avoids tiny clipped labels',foreground='#555').grid(row=3,column=2,sticky='w',padx=6)
        ttk.Checkbutton(chunk_frame,text='Keep empty/negative chunks too',variable=self.train_keep_negative_chunks).grid(row=4,column=0,columnspan=3,sticky='w',pady=2)
        row+=1
        ttk.Checkbutton(tr,text='Auto-resume after crash/interruption',variable=self.auto_resume).grid(row=row,column=0,columnspan=2,sticky='w',pady=4); row+=1
        ttk.Label(tr,text='Projekt').grid(row=row,column=0,sticky='w',pady=4); ttk.Entry(tr,textvariable=self.project).grid(row=row,column=1,sticky='ew',pady=4); row+=1
        tr.columnconfigure(1,weight=1)
        ttk.Button(tr,text='Dependency Check',command=lambda:messagebox.showinfo('Dependency Check',deps())).grid(row=row,column=0,sticky='ew',pady=8)
        ttk.Button(tr,text='Prepare YOLO Dataset',command=lambda:threading.Thread(target=self.prepare_yolo_dataset,daemon=True).start()).grid(row=row,column=1,sticky='ew',pady=8); row+=1
        ttk.Button(tr,text='Train YOLO Model',command=lambda:threading.Thread(target=self.train,daemon=True).start()).grid(row=row,column=0,sticky='ew',pady=8)
        ttk.Button(tr,text='Resume Training from last.pt',command=lambda:threading.Thread(target=lambda:self.train(resume=True),daemon=True).start()).grid(row=row,column=1,sticky='ew',pady=8); row+=1
        ttk.Button(tr,text='Export Best Model to ONNX',command=lambda:threading.Thread(target=self.export_onnx,daemon=True).start()).grid(row=row,column=0,columnspan=2,sticky='ew',pady=4); row+=1
        self.tlog=tk.Text(tr,height=28); self.tlog.grid(row=row,column=0,columnspan=2,sticky='nsew'); tr.rowconfigure(row,weight=1)

        # --- Tab 4: SAM2 Segmentation ---
        self.sam_out=tk.StringVar()
        sam_left=ttk.Frame(t4,padding=10); sam_left.pack(side=tk.LEFT,fill=tk.Y)
        sam_right=ttk.Frame(t4,padding=10); sam_right.pack(fill=tk.BOTH,expand=True)
        ttk.Label(sam_left,text='SAM2 Modell').pack(anchor='w')
        ttk.Entry(sam_left,textvariable=self.sammodel,width=55).pack(fill=tk.X)
        ttk.Button(sam_left,text='SAM2 .pt wählen',command=lambda:self.pick(self.sammodel,[('PT','*.pt'),('All','*.*')])).pack(fill=tk.X)
        ttk.Separator(sam_left).pack(fill=tk.X,pady=8)
        ttk.Label(sam_left,text='Bildordner für SAM2').pack(anchor='w')
        ttk.Entry(sam_left,textvariable=self.sam_source_dir,width=55).pack(fill=tk.X)
        row_btn=ttk.Frame(sam_left); row_btn.pack(fill=tk.X,pady=3)
        ttk.Button(row_btn,text='Ordner wählen',command=lambda:self.pickdir(self.sam_source_dir)).pack(side=tk.LEFT,fill=tk.X,expand=True)
        ttk.Button(row_btn,text='Liste laden',command=self.load_sam_images).pack(side=tk.LEFT,fill=tk.X,expand=True)
        ttk.Button(sam_left,text='aus Crop-Ordner übernehmen',command=self.sam_use_cropdir).pack(fill=tk.X,pady=2)
        ttk.Button(sam_left,text='aus Projekt/images übernehmen',command=self.sam_use_project_images).pack(fill=tk.X,pady=2)
        ttk.Label(sam_left,text='Alle Bilder/Crops').pack(anchor='w',pady=(8,0))
        self.sam_list=tk.Listbox(sam_left,width=55,height=18,exportselection=False)
        self.sam_list.pack(fill=tk.BOTH,expand=True)
        self.sam_list.bind('<<ListboxSelect>>',self.on_sam_select)
        settings=ttk.LabelFrame(sam_left,text='SAM2 Einstellungen',padding=5); settings.pack(fill=tk.X,pady=6)
        ttk.Label(settings,text='Padding um Prompt-Box').grid(row=0,column=0,sticky='w')
        ttk.Entry(settings,textvariable=self.sam_padding,width=10).grid(row=0,column=1,sticky='w')
        ttk.Label(settings,text='Max. Cropgröße').grid(row=1,column=0,sticky='w')
        ttk.Entry(settings,textvariable=self.sam_max_crop,width=10).grid(row=1,column=1,sticky='w')
        ttk.Checkbutton(settings,text='Boxen/Annotationen als Prompts nutzen',variable=self.sam_use_ann_boxes).grid(row=2,column=0,columnspan=2,sticky='w')
        ttk.Checkbutton(settings,text='Batch: bereits segmentierte Bilder überspringen (.sam2.json vorhanden)',variable=self.sam_skip_existing).grid(row=3,column=0,columnspan=2,sticky='w')
        ttk.Label(sam_left,text='Ausgabe JSON').pack(anchor='w',pady=(8,0))
        ttk.Entry(sam_left,textvariable=self.sam_out,width=55).pack(fill=tk.X)
        ttk.Button(sam_left,text='SAM2: ausgewähltes Bild segmentieren',command=lambda:threading.Thread(target=self.sam2_selected,daemon=True).start()).pack(fill=tk.X,pady=4)
        ttk.Button(sam_left,text='SAM2: alle Bilder segmentieren',command=lambda:threading.Thread(target=self.sam2_all,daemon=True).start()).pack(fill=tk.X,pady=4)
        ttk.Label(sam_left,text='Hinweis: Wenn YOLO-Labels vorhanden sind, werden diese Boxen als Prompts genutzt. Sonst wird das ganze Bild als Prompt segmentiert.').pack(anchor='w',pady=4)
        self.sam_canvas=tk.Canvas(sam_right,bg='black',height=620); self.sam_canvas.pack(fill=tk.BOTH,expand=True)
        self.sam_canvas.bind('<Configure>',lambda e:self.draw_sam_preview())
        self.samlog=tk.Text(sam_right,height=10); self.samlog.pack(fill=tk.X)
        self.sam_polys=[]

        # --- Tab 5: FormTrainer ---
        self.form_project=tk.StringVar(value=self.project.get()); self.form_model_path=tk.StringVar(); self.form_epochs=tk.IntVar(value=1200)
        ft=ttk.Frame(t5,padding=10); ft.pack(fill=tk.BOTH,expand=True)
        ttk.Label(ft,text='Trainingsprojekt mit images/ und labels/').grid(row=0,column=0,sticky='w'); ttk.Entry(ft,textvariable=self.form_project,width=85).grid(row=0,column=1,sticky='ew'); ttk.Button(ft,text='...',command=lambda:self.pickdir(self.form_project)).grid(row=0,column=2)
        ttk.Label(ft,text='FormLearner Ausgabe .json').grid(row=1,column=0,sticky='w'); ttk.Entry(ft,textvariable=self.form_model_path,width=85).grid(row=1,column=1,sticky='ew'); ttk.Button(ft,text='...',command=self.pick_form_out).grid(row=1,column=2)
        ttk.Label(ft,text='Epochs').grid(row=2,column=0,sticky='w'); ttk.Entry(ft,textvariable=self.form_epochs,width=10).grid(row=2,column=1,sticky='w')
        ttk.Button(ft,text='FORMTRAINER STARTEN',command=lambda:threading.Thread(target=self.train_formlearner,daemon=True).start()).grid(row=3,column=0,columnspan=3,sticky='ew',pady=5)
        self.formlog=tk.Text(ft); self.formlog.grid(row=4,column=0,columnspan=3,sticky='nsew'); ft.rowconfigure(4,weight=1); ft.columnconfigure(1,weight=1)

        # --- Tab 6: Detection with FormLearner ---
        self.fl_model_path=tk.StringVar(); self.fl_threshold=tk.DoubleVar(value=0.50); self.fl_output=tk.StringVar()
        fl=ttk.Frame(t6,padding=10); fl.pack(fill=tk.BOTH,expand=True)
        fl_left=ttk.Frame(fl); fl_left.pack(side=tk.LEFT,fill=tk.Y)
        fl_right=ttk.Frame(fl); fl_right.pack(side=tk.LEFT,fill=tk.BOTH,expand=True,padx=(8,0))
        ttk.Label(fl_left,text='Nutzt Modelle/Bild/Tile/Overlap aus Tab 1 und filtert anschließend mit FormLearner.').pack(anchor='w')
        ttk.Label(fl_left,text='FormLearner .json').pack(anchor='w',pady=(6,0)); ttk.Entry(fl_left,textvariable=self.fl_model_path,width=62).pack(fill=tk.X); ttk.Button(fl_left,text='FormLearner wählen',command=lambda:self.pick(self.fl_model_path,[('JSON','*.json'),('All','*.*')])).pack(fill=tk.X)
        fr_thr=ttk.Frame(fl_left); fr_thr.pack(fill=tk.X,pady=4)
        ttk.Label(fr_thr,text='Mindest FormScore').pack(side=tk.LEFT); ttk.Scale(fr_thr,variable=self.fl_threshold,from_=0,to=1,orient=tk.HORIZONTAL,command=lambda e:self.draw_fl_preview()).pack(side=tk.LEFT,fill=tk.X,expand=True); ttk.Label(fr_thr,textvariable=self.fl_threshold,width=6).pack(side=tk.LEFT)
        ttk.Label(fl_left,text='Output GeoJSON/GPKG').pack(anchor='w'); ttk.Entry(fl_left,textvariable=self.fl_output,width=62).pack(fill=tk.X)
        ttk.Button(fl_left,text='DETECTION MIT FORM-LEARNER STARTEN',command=lambda:threading.Thread(target=self.detect_with_formlearner,daemon=True).start()).pack(fill=tk.X,pady=5)
        ttk.Button(fl_left,text='Vorschau neu zeichnen',command=self.draw_fl_preview).pack(fill=tk.X,pady=2)
        self.fllog=tk.Text(fl_left,width=62,height=28); self.fllog.pack(fill=tk.BOTH,expand=True)
        ttk.Label(fl_right,text='Vorschau: akzeptierte FormLearner-Detections').pack(anchor='w')
        self.fl_canvas=tk.Canvas(fl_right,bg='black'); self.fl_canvas.pack(fill=tk.BOTH,expand=True)
        self.fl_canvas.bind('<Configure>',lambda e:self.draw_fl_preview())
        self.fl_kept=[]; self.fl_preview_img=None
    def log(self,s=''):
        self.logbox.insert(tk.END,str(s)+'\n'); self.logbox.see(tk.END); self.update_idletasks()
    def tmsg(self,s=''):
        self.tlog.insert(tk.END,str(s)+'\n'); self.tlog.see(tk.END); self.log(s)
    def pick(self,var,types):
        p=filedialog.askopenfilename(filetypes=types)
        if p: var.set(p)
    def pickcrop(self):
        p=filedialog.askdirectory()
        if p: self.cropdir.set(p)
    def pickproj(self):
        p=filedialog.askdirectory()
        if p: self.project.set(p)
    def loadprev(self):
        self.preview,self.origW,self.origH=load_preview(self.image.get()); self.zoom=1; self.pan=[0,0]; self.redraw(); self.log(f'Preview {self.preview.size}, Original {self.origW}x{self.origH}')
    def detect(self):
        try:
            from ultralytics import YOLO
            models=[]
            for i,v in enumerate(self.models):
                p=Path(v.get().strip().strip('"'))
                if p.exists(): self.log(f'Lade Modell {i+1}: {p}'); models.append((i,p.name,YOLO(str(p))))
            if not models: raise RuntimeError('Kein Modell gewählt')
            if len(models)==1:
                self.log('Nur 1 Modell ausgewählt: Detection wird pro Tile genau einmal gerechnet; kein Ensemble-Durchlauf.')
            else:
                self.log(f'{len(models)} Modelle ausgewählt: Ensemble-Farben und Konsensbewertung aktiv.')
            img=Path(self.image.get().strip().strip('"')); self.last_img=img; W,H,mode,reader=open_img(img,self.log); self.last_geo=('pixel',None,None)
            if mode=='rasterio':
                try: self.last_geo=('rasterio',reader.transform,reader.crs.to_string() if reader.crs else None)
                except Exception: pass
            tile=int(self.tile.get()); overlap=int(self.overlap.get()); pts=positions(W,H,tile,overlap,self.shift.get()); self.dets=[]; t0=time.time()
            for n,(x,y) in enumerate(pts,1):
                im=read_tile(reader,mode,x,y,tile,W,H); arr=np.asarray(im)
                for slot,name,m in models:
                    try:
                        res=m.predict(arr,conf=float(self.conf.get()),imgsz=tile,verbose=False)
                        if res and res[0].boxes is not None:
                            b=res[0].boxes; xy=b.xyxy.cpu().numpy(); cf=b.conf.cpu().numpy(); cl=b.cls.cpu().numpy()
                            for bb,c,k in zip(xy,cf,cl):
                                x1,y1,x2,y2=map(float,bb[:4])
                                if x2>x1 and y2>y1: self.dets.append(Det(slot,name,int(k),float(c),x+x1,y+y1,x+x2,y+y2))
                    except Exception as e: self.log(f'Modell {slot+1} Tile {x},{y}: {e}')
                if n%10==0: self.log(f'Stand {n}/{len(pts)} Tiles | Detections={len(self.dets)} | {n/max(.1,time.time()-t0):.2f} tiles/s')
            if mode=='rasterio': reader.close()
            self.score(); self.log(f'Detection fertig: {len(self.dets)} Treffer'); self.redraw()
        except Exception as e: self.log('ERROR '+str(e)); messagebox.showerror(APP,str(e))
    def score(self):
        used=set()
        for i,d in enumerate(self.dets):
            if i in used: continue
            group=[i]; used.add(i)
            for j,e in enumerate(self.dets):
                if j not in used and any(iou(e.bbox(),self.dets[k].bbox())>.25 for k in group): group.append(j); used.add(j)
            cons=len(set(self.dets[k].slot for k in group)); sc=cons+sum(self.dets[k].conf for k in group)/len(group)
            for k in group: self.dets[k].consensus=cons; self.dets[k].score=sc
    def visible(self):
        # Nur Vorschau: Anzeige-Confidence darf keine Export-/FormLearner-Ergebnisse verändern.
        return [d for d in self.dets if d.conf>=float(self.showconf.get()) and d.score>=float(self.minscore.get())]
    def computed_candidates(self):
        # Für Export/FormLearner: alle bereits berechneten Treffer, nicht der Anzeige-Confidence-Slider.
        return [d for d in self.dets if d.score>=float(self.minscore.get())]
    def redraw(self):
        if self.preview is None: return
        self.can.delete('all'); W,H=self.preview.size; sc=min(self.can.winfo_width()/W,self.can.winfo_height()/H)*self.zoom; sc=max(.02,sc); nw=int(W*sc); nh=int(H*sc); img=self.preview.resize((nw,nh)); self.ph=ImageTk.PhotoImage(img); x0=(self.can.winfo_width()-nw)//2+self.pan[0]; y0=(self.can.winfo_height()-nh)//2+self.pan[1]; self.can.create_image(x0,y0,anchor='nw',image=self.ph)
        sx=nw/self.origW; sy=nh/self.origH
        for d in self.visible():
            colors=['lime','cyan','magenta','yellow','orange','red']; col=colors[d.slot%len(colors)]
            if d.consensus>1: col='white'
            self.can.create_rectangle(x0+d.x1*sx,y0+d.y1*sy,x0+d.x2*sx,y0+d.y2*sy,outline=col,width=2)
            self.can.create_text(x0+d.x1*sx,y0+d.y1*sy,anchor='sw',fill=col,text=f'M{d.slot+1} {d.conf:.2f}/s{d.score:.1f}')
    def wheel(self,e): self.zoom*=1.15 if e.delta>0 else 1/1.15; self.redraw()
    def press(self,e): self.pp=(e.x,e.y,self.pan[0],self.pan[1])
    def move(self,e): x,y,px,py=self.pp; self.pan=[px+e.x-x,py+e.y-y]; self.redraw()
    def export_crops(self):
        """Export visible detections as neutral 1024x1024 PNG crops.
        No label files are written here. The annotator later writes class 0/1 labels per image.
        """
        try:
            img=Path(getattr(self,'last_img',Path(self.image.get())))
            out=Path(self.cropdir.get() or str(img.with_name(img.stem+'_training_crops')))
            self.cropdir.set(str(out))
            (out/'images').mkdir(parents=True,exist_ok=True)
            (out/'labels').mkdir(exist_ok=True)
            (out/'project.json').write_text(json.dumps({'classes':CLASSES,'annotation_mode':'neutral_until_review'},indent=2),encoding='utf-8')
            W,H,mode,reader=open_img(img,self.log)
            n=0; skipped_overlap=0; meta=[]; size=int(self.cropsize.get()); pad=int(self.pad.get())

            def rect_intersects(a,b):
                ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
                return max(ax1,bx1) < min(ax2,bx2) and max(ay1,by1) < min(ay2,by2)

            # Exported crop windows must not overlap.
            # Crop export intentionally follows the current preview filter:
            # Anzeige-Confidence + Mindestscore decide what gets exported.
            # Keep strongest visible candidates first, then skip any crop window that intersects an already saved crop.
            candidates=[]
            for d in self.visible():
                x1,y1,x2,y2=d.bbox(); cx=(x1+x2)/2; cy=(y1+y2)/2
                side=max(size,int(max(x2-x1,y2-y1)+2*pad))
                left=int(max(0,min(W-side,cx-side/2))); top=int(max(0,min(H-side,cy-side/2)))
                right=min(W,left+side); bottom=min(H,top+side)
                rank=(float(getattr(d,'score',0.0)), float(getattr(d,'conf',0.0)), float((x2-x1)*(y2-y1)))
                candidates.append((rank,d,left,top,right,bottom,side))
            candidates.sort(key=lambda t:t[0], reverse=True)

            saved_rects=[]
            for _rank,d,left,top,right,bottom,side in candidates:
                crop_rect=(left,top,right,bottom)
                if any(rect_intersects(crop_rect,r) for r in saved_rects):
                    skipped_overlap+=1
                    continue
                x1,y1,x2,y2=d.bbox()
                crop=read_tile(reader,mode,left,top,side,W,H)
                sx=size/crop.width; sy=size/crop.height
                crop_bbox=[max(0,min(size,(x1-left)*sx)),max(0,min(size,(y1-top)*sy)),max(0,min(size,(x2-left)*sx)),max(0,min(size,(y2-top)*sy))]
                if crop.size!=(size,size):
                    crop=crop.resize((size,size))
                name=f'{img.stem}_det{n:06d}_m{d.slot+1}_conf{d.conf:.3f}_x{left}_y{top}'.replace('.','p')
                crop.save(out/'images'/(name+'.png'))
                saved_rects.append(crop_rect)
                # Important: do NOT write label here. Crop starts neutral/unreviewed.
                meta.append({'image':name+'.png','status':'neutral','bbox_global':d.bbox(),'bbox_crop':crop_bbox,'conf':d.conf,'score':getattr(d,'score',0.0),'model':d.name,'model_slot':d.slot+1,'crop_left':left,'crop_top':top,'crop_right':right,'crop_bottom':bottom,'crop_size':size})
                n+=1
                if n%25==0: self.log(f'Neutrale, nicht überlappende Crops {n}')
            if skipped_overlap:
                self.log(f'Überlappende Crops übersprungen: {skipped_overlap}')
            if mode=='rasterio': reader.close()
            (out/'crop_manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
            self.project.set(str(out))
            self.log(f'{n} neutrale PNG-Crops gespeichert: {out}')
            self.log(f'Crop-Export nutzte Anzeige-Confidence >= {float(self.showconf.get()):.3f} und Mindestscore >= {float(self.minscore.get()):.3f}')
            self.log('Bitte im Crop-Annotator jedes Bild als POSITIV oder FALSE markieren.')
        except Exception as e:
            self.log('Crop error '+str(e)); messagebox.showerror(APP,str(e))

    def loadproj(self):
        root=Path(self.project.get())
        (root/'labels').mkdir(exist_ok=True)
        self.ann_imgs=sorted([p for p in (root/'images').iterdir() if p.suffix.lower() in IMG_EXT])
        self.ann_i=0
        self.ann_manifest={}
        mf=root/'crop_manifest.json'
        if mf.exists():
            try:
                data=json.loads(mf.read_text(encoding='utf-8'))
                self.ann_manifest={d.get('image'):d for d in data if d.get('image')}
            except Exception as e:
                self.log('Manifest konnte nicht gelesen werden: '+str(e))
        self.showann()

    def current_label_path(self):
        if not self.ann_imgs: return None
        return Path(self.project.get())/'labels'/(self.ann_imgs[self.ann_i].stem+'.txt')

    def showann(self):
        if not self.ann_imgs:
            if hasattr(self,'ann_status'): self.ann_status.set('Keine Bilder im Projektordner/images gefunden.')
            return
        p=self.ann_imgs[self.ann_i]
        self.aimg=Image.open(p).convert('RGB')
        lp=self.current_label_path()
        self.ann_boxes=read_boxes(p,lp) if lp and lp.exists() else []
        self.ann_selected=0 if self.ann_boxes else -1
        self.drawann()
        self.refresh_ann_list()
        status='NEUTRAL / ungeprüft' if not self.ann_boxes else ('POSITIV' if any(int(b[0])==0 for b in self.ann_boxes) else 'FALSE')
        self.ann_status.set(f'{self.ann_i+1}/{len(self.ann_imgs)}  {p.name}\nStatus: {status}\nBoxen: {len(self.ann_boxes)}')
        self.title(f'{APP} | {self.ann_i+1}/{len(self.ann_imgs)} {p.name}')

    def ann_transform(self):
        W,H=self.aimg.size
        cw=max(10,self.acan.winfo_width()); ch=max(10,self.acan.winfo_height())
        base=min(cw/W,ch/H)
        sc=max(0.02,base*getattr(self,'ann_zoom',1.0))
        nw=int(W*sc); nh=int(H*sc)
        ax=(cw-nw)//2+getattr(self,'ann_pan',[0,0])[0]
        ay=(ch-nh)//2+getattr(self,'ann_pan',[0,0])[1]
        return sc,ax,ay,nw,nh

    def drawann(self):
        if not hasattr(self,'aimg'): return
        self.acan.delete('all')
        sc,ax,ay,nw,nh=self.ann_transform(); self.asc=sc; self.ax=ax; self.ay=ay
        self.aph=ImageTk.PhotoImage(self.aimg.resize((max(1,nw),max(1,nh))))
        self.acan.create_image(ax,ay,anchor='nw',image=self.aph)
        # Optional SAM2 masks for this crop are shown as cyan polygons under the annotation boxes.
        try:
            curp=self.ann_imgs[self.ann_i] if self.ann_imgs else None
            if curp:
                side=curp.with_suffix('.sam2.json')
                if side.exists():
                    data=json.loads(side.read_text(encoding='utf-8'))
                    items=data if isinstance(data,list) else data.get('polygons',[])
                    for item in items:
                        pts=item.get('polygon') if isinstance(item,dict) else None
                        if pts and len(pts)>=2:
                            flat=[]
                            for x,y in pts: flat.extend([ax+x*sc,ay+y*sc])
                            if len(flat)>=4: self.acan.create_line(*flat,fill='cyan',width=2)
        except Exception:
            pass
        for idx,(c,x1,y1,x2,y2) in enumerate(self.ann_boxes):
            col='lime' if int(c)==0 else 'red'
            width=4 if idx==getattr(self,'ann_selected',-1) else 2
            if idx==getattr(self,'ann_selected',-1): col='orange'
            self.acan.create_rectangle(ax+x1*sc,ay+y1*sc,ax+x2*sc,ay+y2*sc,outline=col,width=width)
            label='POS' if int(c)==0 else 'FALSE'
            self.acan.create_text(ax+x1*sc+3,ay+y1*sc+3,anchor='nw',fill=col,text=f'{idx+1}: {label}')

    def refresh_ann_list(self):
        if not hasattr(self,'ann_list'): return
        self.ann_list.delete(0,tk.END)
        for i,b in enumerate(self.ann_boxes):
            c,x1,y1,x2,y2=b
            name='POSITIV / mustatil' if int(c)==0 else 'FALSE positive'
            self.ann_list.insert(tk.END,f'{i+1:02d} | {name} | x={x1:.0f} y={y1:.0f} w={x2-x1:.0f} h={y2-y1:.0f}')
        if 0 <= getattr(self,'ann_selected',-1) < len(self.ann_boxes):
            self.ann_list.selection_set(self.ann_selected)

    def on_ann_select(self,event=None):
        if not hasattr(self,'ann_list'): return
        sel=self.ann_list.curselection()
        self.ann_selected=int(sel[0]) if sel else -1
        self.drawann()

    def manifest_bbox_for_current(self):
        if not self.ann_imgs: return None
        p=self.ann_imgs[self.ann_i]
        m=getattr(self,'ann_manifest',{}).get(p.name,{})
        bb=m.get('bbox_crop')
        if bb and len(bb)>=4:
            return [float(bb[0]),float(bb[1]),float(bb[2]),float(bb[3])]
        W,H=self.aimg.size
        pad=max(20,int(min(W,H)*0.12))
        return [pad,pad,W-pad,H-pad]

    def set_image_class(self,cls):
        if not self.ann_imgs: return
        bb=self.manifest_bbox_for_current()
        self.ann_boxes=[[int(cls),bb[0],bb[1],bb[2],bb[3]]]
        self.ann_selected=0
        self.saveann()
        self.drawann(); self.refresh_ann_list(); self.showann()

    def set_selected_cls(self):
        if 0 <= getattr(self,'ann_selected',-1) < len(self.ann_boxes):
            self.ann_boxes[self.ann_selected][0]=int(self.ann_cls.get())
            self.saveann(); self.drawann(); self.refresh_ann_list()

    def delete_selected_box(self):
        if 0 <= getattr(self,'ann_selected',-1) < len(self.ann_boxes):
            del self.ann_boxes[self.ann_selected]
            self.ann_selected=min(self.ann_selected,len(self.ann_boxes)-1)
            self.saveann(); self.drawann(); self.refresh_ann_list(); self.showann()

    def clear_current_ann(self):
        self.ann_boxes=[]; self.ann_selected=-1; self.saveann(); self.drawann(); self.refresh_ann_list(); self.showann()

    def awheel(self,e):
        self.ann_zoom*=1.15 if e.delta>0 else 1/1.15
        self.ann_zoom=max(0.1,min(20,self.ann_zoom)); self.drawann()

    def apan_press(self,e):
        self.ann_pan_drag=(e.x,e.y,self.ann_pan[0],self.ann_pan[1])
    def apan_move(self,e):
        x,y,px,py=self.ann_pan_drag
        self.ann_pan=[px+e.x-x,py+e.y-y]
        self.drawann()

    def apress(self,e):
        self.drag=(e.x,e.y)

    def arelease(self,e):
        if not self.drag: return
        x0,y0=self.drag; x1,y1=e.x,e.y; sc=self.asc
        bx1=(min(x0,x1)-self.ax)/sc; by1=(min(y0,y1)-self.ay)/sc
        bx2=(max(x0,x1)-self.ax)/sc; by2=(max(y0,y1)-self.ay)/sc
        W,H=self.aimg.size
        bx1=max(0,min(W,bx1)); bx2=max(0,min(W,bx2)); by1=max(0,min(H,by1)); by2=max(0,min(H,by2))
        if bx2-bx1>5 and by2-by1>5:
            self.ann_boxes.append([int(self.ann_cls.get()),bx1,by1,bx2,by2])
            self.ann_selected=len(self.ann_boxes)-1
            self.saveann(); self.drawann(); self.refresh_ann_list(); self.showann()
        self.drag=None

    def setcls(self):
        # Kept for compatibility with older buttons/scripts.
        self.set_selected_cls()

    def saveann(self):
        if not self.ann_imgs: return
        p=self.ann_imgs[self.ann_i]
        lp=self.current_label_path()
        if not self.ann_boxes:
            # Neutral/unreviewed: remove label file instead of writing an empty one.
            try:
                if lp and lp.exists(): lp.unlink()
            except Exception: pass
            return
        save_boxes(p,lp,self.ann_boxes)

    def nextann(self,d):
        self.saveann()
        if not self.ann_imgs: return
        self.ann_i=max(0,min(len(self.ann_imgs)-1,self.ann_i+d))
        self.ann_selected=-1; self.ann_zoom=1.0; self.ann_pan=[0,0]
        self.showann()

    def yolo_clip_box_for_tile(self, box, tx, ty, tw, th, min_visible):
        c,x1,y1,x2,y2=box
        ox1,ox2=sorted((float(x1),float(x2))); oy1,oy2=sorted((float(y1),float(y2)))
        area=max(1.0,(ox2-ox1)*(oy2-oy1))
        ix1=max(ox1,tx); iy1=max(oy1,ty); ix2=min(ox2,tx+tw); iy2=min(oy2,ty+th)
        if ix2<=ix1 or iy2<=iy1: return None
        if ((ix2-ix1)*(iy2-iy1))/area < float(min_visible): return None
        return [int(c), ix1-tx, iy1-ty, ix2-tx, iy2-ty]

    def save_yolo_boxes_tile(self, label_path, boxes, w, h):
        lines=[]
        for c,x1,y1,x2,y2 in boxes:
            x1,x2=sorted((max(0,x1),min(w,x2))); y1,y2=sorted((max(0,y1),min(h,y2)))
            if x2>x1 and y2>y1:
                lines.append(f'{int(c)} {((x1+x2)/2/w):.8f} {((y1+y2)/2/h):.8f} {((x2-x1)/w):.8f} {((y2-y1)/h):.8f}')
        Path(label_path).write_text('\n'.join(lines)+('\n' if lines else ''),encoding='utf-8')

    def prepare_yolo_dataset(self):
        root=Path(self.project.get())
        all_imgs=sorted([p for p in (root/'images').iterdir() if p.suffix.lower() in IMG_EXT])
        labs=root/'labels'
        imgs=[p for p in all_imgs if (labs/(p.stem+'.txt')).exists()]
        if not imgs: raise RuntimeError('Keine geprüften Annotationen gefunden. Bitte im Crop-Annotator POSITIV/FALSE klicken.')
        ds=root/'_yolo_dataset'; shutil.rmtree(ds,ignore_errors=True)
        for sub in ['images/train','images/val','labels/train','labels/val']:
            (ds/sub).mkdir(parents=True,exist_ok=True)
        split=max(1,int(len(imgs)*.8)); tr=imgs[:split]; va=imgs[split:] or imgs[:1]
        chunk_enabled=bool(self.train_chunk_enabled.get()); chunk_size=int(self.train_chunk_size.get()); chunk_overlap=int(self.train_chunk_overlap.get())
        min_visible=float(self.train_chunk_min_visible.get()); keep_negative=bool(self.train_keep_negative_chunks.get())
        self.tmsg(f'Training chunks active: {chunk_enabled}; chunk={chunk_size}; overlap={chunk_overlap}; min_visible={min_visible}; keep_negative={keep_negative}')
        total=0; skipped=0
        def add_image(p, subset):
            nonlocal total, skipped
            boxes=read_boxes(p,labs/(p.stem+'.txt'))
            if not chunk_enabled:
                shutil.copy2(p,ds/f'images/{subset}/{p.name}'); shutil.copy2(labs/(p.stem+'.txt'),ds/f'labels/{subset}/{p.stem}.txt'); total+=1; return
            W,H,mode,reader=open_img(p,self.tmsg); stride=max(1,chunk_size-chunk_overlap)
            try:
                for y in range(0,H,stride):
                    for x in range(0,W,stride):
                        tw=min(chunk_size,W-x); th=min(chunk_size,H-y)
                        if tw<64 or th<64: continue
                        tboxes=[]
                        for b in boxes:
                            cb=self.yolo_clip_box_for_tile(b,x,y,tw,th,min_visible)
                            if cb is not None: tboxes.append(cb)
                        if not tboxes and not keep_negative:
                            skipped+=1; continue
                        crop=read_tile(reader,mode,x,y,chunk_size,W,H).convert('RGB')
                        name=f'{p.stem}_x{x}_y{y}_w{tw}_h{th}.jpg'
                        crop.save(ds/f'images/{subset}/{name}',quality=95,subsampling=0)
                        self.save_yolo_boxes_tile(ds/f'labels/{subset}/{Path(name).stem}.txt',tboxes,tw,th)
                        total+=1
                    if y+chunk_size>=H: break
            finally:
                try:
                    if mode=='rasterio': reader.close()
                except Exception: pass
        for p in tr: add_image(p,'train')
        for p in va: add_image(p,'val')
        yml=ds/'data.yaml'; yml.write_text(f"path: {ds.as_posix()}\ntrain: images/train\nval: images/val\nnc: 2\nnames: ['mustatil','false_positive']\n",encoding='utf-8')
        self.tmsg(f'Total YOLO training images/chunks: {total}; skipped empty chunks: {skipped}')
        self.tmsg(f'Dataset ready: {yml}')
        return yml

    def export_onnx(self):
        try:
            root=Path(self.project.get()); best=root/'runs'/'train_mustatil'/'weights'/'best.pt'
            if not best.exists(): raise RuntimeError(f'best.pt nicht gefunden: {best}')
            cmd=[sys.executable,'-c',"from ultralytics import YOLO; import sys; YOLO(sys.argv[1]).export(format='onnx', imgsz=int(sys.argv[2]))",str(best),str(self.imgsz.get())]
            run_live(cmd,self.tmsg,root)
        except Exception as e:
            self.tmsg('ONNX EXPORT ERROR '+str(e)); messagebox.showerror(APP,str(e))

    def train(self, resume=False):
        try:
            root=Path(self.project.get())
            y=self.prepare_yolo_dataset()
            imgsz=int(self.imgsz.get()); batch=int(self.batch.get()); device=(self.device.get().strip() or 'cpu').lower()
            if self.low_ram_mode.get():
                imgsz=min(imgsz,640); batch=min(batch,2); self.tmsg('Low-RAM stabil mode active: imgsz<=640, batch<=2, workers=0, cache=False, plots=False.')
            extra=''
            if resume:
                last=root/'runs'/'train_mustatil'/'weights'/'last.pt'
                if last.exists(): model_arg=str(last); extra=', resume=True'
                else: model_arg=self.trainmodel.get(); self.tmsg('last.pt nicht gefunden, starte normales Training.')
            else:
                model_arg=self.trainmodel.get()
            code=("from ultralytics import YOLO\nimport sys\n"
                  "model=YOLO(sys.argv[1])\n"
                  f"model.train(data=r'{str(y)}',epochs={int(self.epochs.get())},imgsz={imgsz},batch={batch},device=r'{device}',project=r'{str(root/'runs')}',name='train_mustatil',exist_ok=True,workers=0,cache=False,plots=False{extra})\n")
            run_live([sys.executable,'-c',code,model_arg],self.tmsg,root)
            self.tmsg('Training complete. Best model is usually runs/train_mustatil/weights/best.pt')
        except Exception as e:
            self.tmsg('TRAIN ERROR '+str(e)); messagebox.showerror(APP,str(e))
    def sam_use_cropdir(self):
        if self.cropdir.get().strip():
            self.sam_source_dir.set(self.cropdir.get().strip())
        self.load_sam_images()

    def sam_use_project_images(self):
        root=Path(self.project.get().strip() or '.')
        self.sam_source_dir.set(str(root/'images'))
        self.load_sam_images()

    def load_sam_images(self):
        folder=Path(self.sam_source_dir.get().strip().strip('"') or self.cropdir.get().strip().strip('"') or '.')
        if not folder.exists():
            messagebox.showerror(APP,f'Bildordner nicht gefunden: {folder}')
            return
        self.sam_images=sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT])
        self.sam_list.delete(0,tk.END)
        for p in self.sam_images:
            self.sam_list.insert(tk.END,p.name)
        self.sammsg(f'SAM2 Bildliste geladen: {len(self.sam_images)} Bilder aus {folder}')
        if self.sam_images:
            self.sam_list.selection_set(0)
            self.on_sam_select()

    def on_sam_select(self,event=None):
        sel=self.sam_list.curselection() if hasattr(self,'sam_list') else []
        if not sel: return
        idx=int(sel[0])
        if idx<0 or idx>=len(self.sam_images): return
        self.sam_selected_img=self.sam_images[idx]
        self.sam_preview_img=self.sam_selected_img
        self.sam_current_polys=[]
        # Load existing sidecar segmentation if present.
        side=self.sam_selected_img.with_suffix('.sam2.json')
        if side.exists():
            try:
                data=json.loads(side.read_text(encoding='utf-8'))
                self.sam_current_polys=data if isinstance(data,list) else data.get('polygons',[])
            except Exception:
                self.sam_current_polys=[]
        self.sam_polys=self.sam_current_polys
        try:
            with Image.open(self.sam_selected_img) as _im:
                _W,_H=_im.size
            self.sam_prompt_boxes=self.sam_boxes_for_image(Path(self.sam_selected_img),_W,_H)
        except Exception:
            self.sam_prompt_boxes=[]
        self.draw_sam_preview()
        self.sammsg(f'Ausgewählt: {self.sam_selected_img.name} | Prompt-Boxen={len(self.sam_prompt_boxes)}')

    def sam_boxes_for_image(self,img:Path,W:int,H:int):
        """Return prompt boxes for SAM2. Prefer YOLO labels next to project labels; fallback whole image."""
        boxes=[]
        # 0) Current crop annotator boxes, if this image is currently open there.
        try:
            if bool(self.sam_use_ann_boxes.get()) and self.ann_imgs and Path(self.ann_imgs[self.ann_i]).resolve()==img.resolve() and self.ann_boxes:
                return [[float(b[1]),float(b[2]),float(b[3]),float(b[4])] for b in self.ann_boxes]
        except Exception:
            pass
        # 1) YOLO label in same folder/../labels
        candidates=[img.with_suffix('.txt'), img.parent.parent/'labels'/(img.stem+'.txt')]
        # 2) labels from current project
        try:
            root=Path(self.project.get().strip())
            candidates.append(root/'labels'/(img.stem+'.txt'))
        except Exception:
            pass
        for lab in candidates:
            try:
                if lab.exists():
                    for line in lab.read_text(encoding='utf-8',errors='ignore').splitlines():
                        parts=line.split()
                        if len(parts)>=5:
                            cls=float(parts[0]); cx,cy,bw,bh=map(float,parts[1:5])
                            x1=(cx-bw/2)*W; y1=(cy-bh/2)*H; x2=(cx+bw/2)*W; y2=(cy+bh/2)*H
                            boxes.append([max(0,x1),max(0,y1),min(W-1,x2),min(H-1,y2)])
                    if boxes:
                        return boxes
            except Exception:
                pass
        # 3) If this is the current large detection image, use visible detections.
        try:
            cur=Path(self.image.get().strip())
            if cur.exists() and cur.resolve()==img.resolve() and self.dets:
                return [list(d.bbox()) for d in self.visible()]
        except Exception:
            pass
        # 4) Fallback: full image prompt, useful for 1024 crop images.
        return [[0,0,W-1,H-1]]

    def sam2_segment_image(self,img:Path,sam=None,save=True):
        from ultralytics import SAM
        if sam is None:
            sam=SAM(self.sammodel.get().strip() or 'sam2_b.pt')
        W,H,mode,reader=open_img(img,self.log)
        boxes=self.sam_boxes_for_image(img,W,H)
        polys=[]
        self.sammsg(f'SAM2 Start: {img.name} | Prompts={len(boxes)} | Größe={W}x{H}')
        try:
            for i,bb in enumerate(boxes,1):
                x1,y1,x2,y2=map(float,bb)
                pad=int(self.sam_padding.get())
                left=max(0,int(x1-pad)); top=max(0,int(y1-pad))
                right=min(W,int(x2+pad)); bottom=min(H,int(y2+pad))
                tw=max(1,right-left); th=max(1,bottom-top)
                crop=read_tile(reader,mode,left,top,max(tw,th),W,H)
                rb=[x1-left,y1-top,x2-left,y2-top]
                scale=1.0
                max_side=max(crop.size)
                max_crop=max(128,int(self.sam_max_crop.get()))
                if max_side>max_crop:
                    scale=max_crop/float(max_side)
                    crop=crop.resize((max(1,int(crop.width*scale)),max(1,int(crop.height*scale))))
                    rb=[v*scale for v in rb]
                try:
                    res=sam.predict(np.asarray(crop),bboxes=[rb],verbose=False)
                    if res and res[0].masks is not None and getattr(res[0].masks,'xy',None):
                        inv=1.0/scale
                        poly=[(float(x)*inv+left,float(y)*inv+top) for x,y in res[0].masks.xy[0]]
                        polys.append({'image':img.name,'bbox':[x1,y1,x2,y2],'polygon':poly,'prompt_index':i,'padding':pad,'max_crop':max_crop})
                except Exception as e:
                    self.sammsg(f'SAM2 Fehler {img.name} Prompt {i}: {e}')
                if i%5==0 or i==len(boxes):
                    self.sammsg(f'SAM2 Stand {img.name}: {i}/{len(boxes)} | Masken={len(polys)}')
        finally:
            try:
                if mode=='rasterio': reader.close()
            except Exception: pass
        if save:
            side=img.with_suffix('.sam2.json')
            side.write_text(json.dumps(polys,indent=2),encoding='utf-8')
            self.sammsg(f'SAM2 gespeichert: {side}')
        return polys

    def sam2_selected(self):
        try:
            if not getattr(self,'sam_selected_img',None):
                self.load_sam_images()
            if not getattr(self,'sam_selected_img',None):
                raise RuntimeError('Kein Bild ausgewählt.')
            polys=self.sam2_segment_image(Path(self.sam_selected_img),sam=None,save=True)
            self.sam_polys=polys; self.sam_current_polys=polys; self.sam_preview_img=Path(self.sam_selected_img)
            out=Path(self.sam_out.get().strip() or str(Path(self.sam_selected_img).with_suffix('.sam2.json')))
            self.sam_out.set(str(out))
            if out != Path(self.sam_selected_img).with_suffix('.sam2.json'):
                out.write_text(json.dumps(polys,indent=2),encoding='utf-8')
            self.draw_sam_preview()
            self.sammsg(f'SAM2 fertig für ausgewähltes Bild: {Path(self.sam_selected_img).name}')
        except Exception as e:
            self.sammsg('SAM2 ERROR '+str(e)); messagebox.showerror(APP,str(e))

    def sam2_all(self):
        try:
            if not getattr(self,'sam_images',None): self.load_sam_images()
            if not self.sam_images: raise RuntimeError('Keine Bilder in der SAM2-Liste.')
            from ultralytics import SAM
            sam=SAM(self.sammodel.get().strip() or 'sam2_b.pt')
            all_polys=[]
            skipped=0
            for idx,img in enumerate(self.sam_images,1):
                imgp=Path(img)
                if self.sam_skip_existing.get() and imgp.with_suffix('.sam2.json').exists():
                    skipped += 1
                    self.sammsg(f'=== SAM2 übersprungen, schon vorhanden: {idx}/{len(self.sam_images)} {imgp.name} ===')
                    continue
                self.sammsg(f'=== SAM2 alle Bilder: {idx}/{len(self.sam_images)} ===')
                polys=self.sam2_segment_image(imgp,sam=sam,save=True)
                all_polys.extend(polys)
                self.sam_polys=polys; self.sam_current_polys=polys; self.sam_preview_img=imgp; self.draw_sam_preview()
            if skipped:
                self.sammsg(f'SAM2 Batch: bereits segmentierte Bilder übersprungen: {skipped}')
            out=Path(self.sam_out.get().strip() or str(Path(self.sam_source_dir.get()).joinpath('sam2_all_polygons.json')))
            self.sam_out.set(str(out)); out.write_text(json.dumps(all_polys,indent=2),encoding='utf-8')
            self.sammsg(f'SAM2 alle Bilder fertig: {len(all_polys)} Masken | Sammeldatei: {out}')
        except Exception as e:
            self.sammsg('SAM2 ALL ERROR '+str(e)); messagebox.showerror(APP,str(e))

    # Backward compatible button target: segment selected/current detection image.
    def sam2(self):
        return self.sam2_selected()

    def sam2_current_annotation(self):
        try:
            if not self.ann_imgs:
                raise RuntimeError('Kein Crop-Projekt geladen.')
            img=Path(self.ann_imgs[self.ann_i])
            self.sam_selected_img=img; self.sam_preview_img=img
            self.sam_source_dir.set(str(img.parent))
            polys=self.sam2_segment_image(img,sam=None,save=True)
            self.sam_polys=polys; self.sam_current_polys=polys
            with Image.open(img) as _im:
                self.sam_prompt_boxes=self.sam_boxes_for_image(img,_im.width,_im.height)
            self.draw_sam_preview()
            self.drawann()
            self.sammsg(f'SAM2 aktuelles Crop fertig: {img.name}')
        except Exception as e:
            self.sammsg('SAM2 CROP ERROR '+str(e)); messagebox.showerror(APP,str(e))

    def sam2_all_annotation_project(self):
        try:
            if not self.ann_imgs:
                self.loadproj()
            if not self.ann_imgs:
                raise RuntimeError('Keine Crops im Projekt geladen.')
            from ultralytics import SAM
            sam=SAM(self.sammodel.get().strip() or 'sam2_b.pt')
            skipped=0
            for i,img in enumerate(self.ann_imgs,1):
                imgp=Path(img)
                if self.sam_skip_existing.get() and imgp.with_suffix('.sam2.json').exists():
                    skipped += 1
                    self.sammsg(f'SAM2 Projekt-Crop übersprungen, schon vorhanden: {i}/{len(self.ann_imgs)} {imgp.name}')
                    continue
                self.sammsg(f'SAM2 Projekt-Crops: {i}/{len(self.ann_imgs)} {imgp.name}')
                self.sam2_segment_image(imgp,sam=sam,save=True)
            if skipped:
                self.sammsg(f'SAM2 Projekt-Crops übersprungen: {skipped}')
            self.showann()
            self.sammsg('SAM2 für alle Projekt-Crops fertig.')
        except Exception as e:
            self.sammsg('SAM2 ALL CROPS ERROR '+str(e)); messagebox.showerror(APP,str(e))

    def draw_sam_preview(self):
        try:
            if not hasattr(self,'sam_canvas') or not getattr(self,'sam_preview_img',None): return
            img,OW,OH=load_preview(self.sam_preview_img,1600)
            cw=max(10,self.sam_canvas.winfo_width()); ch=max(10,self.sam_canvas.winfo_height())
            sc=min(cw/img.width,ch/img.height); nw=max(1,int(img.width*sc)); nh=max(1,int(img.height*sc))
            im=img.resize((nw,nh)); self.sam_photo=ImageTk.PhotoImage(im)
            self.sam_canvas.delete('all'); x0=(cw-nw)//2; y0=(ch-nh)//2
            self.sam_canvas.create_image(x0,y0,anchor='nw',image=self.sam_photo)
            sx=nw/OW; sy=nh/OH
            # Prompt/annotation boxes are shown even before SAM2 runs.
            for bb in getattr(self,'sam_prompt_boxes',[]):
                try:
                    x1,y1,x2,y2=bb
                    self.sam_canvas.create_rectangle(x0+x1*sx,y0+y1*sy,x0+x2*sx,y0+y2*sy,outline='orange',width=2)
                    self.sam_canvas.create_text(x0+x1*sx+3,y0+y1*sy+3,anchor='nw',fill='orange',text='Prompt/Anno')
                except Exception:
                    pass
            for item in getattr(self,'sam_polys',[]):
                pts=item.get('polygon') or []
                if len(pts)>=2:
                    flat=[]
                    for x,y in pts: flat.extend([x0+x*sx,y0+y*sy])
                    if len(flat)>=4: self.sam_canvas.create_line(*flat,fill='cyan',width=2)
                bb=item.get('bbox')
                if bb:
                    x1,y1,x2,y2=bb; self.sam_canvas.create_rectangle(x0+x1*sx,y0+y1*sy,x0+x2*sx,y0+y2*sy,outline='yellow',width=1)
        except Exception as e:
            try: self.sammsg('SAM Preview Fehler: '+str(e))
            except Exception: pass

    def export_geo(self):
        out=Path(self.output.get() or str(Path(self.image.get()).with_suffix('.detections.geojson'))); feats=[]
        for i,d in enumerate(self.visible(),1):
            x1,y1,x2,y2=d.bbox(); feats.append({'type':'Feature','geometry':{'type':'Polygon','coordinates':[[(x1,y1),(x2,y1),(x2,y2),(x1,y2),(x1,y1)]]},'properties':{'rank':i,'model':d.name,'model_index':d.slot+1,'confidence':d.conf,'score':d.score,'consensus':d.consensus,'pixel_bbox':[x1,y1,x2,y2]}})
        if out.suffix.lower()=='.gpkg':
            try:
                write_gpkg_fallback(out, feats, layer='detections', crs_name=None, log_fn=self.log)
            except Exception as e:
                self.log('GPKG fehlgeschlagen, schreibe GeoJSON: '+str(e)); out=out.with_suffix('.geojson'); out.write_text(json.dumps({'type':'FeatureCollection','features':feats},indent=2),encoding='utf-8')
        else: out.write_text(json.dumps({'type':'FeatureCollection','features':feats},indent=2),encoding='utf-8')
        self.log(f'Export: {out}')
    def pickdir(self,var):
        p=filedialog.askdirectory()
        if p: var.set(p)
    def pick_form_out(self):
        p=filedialog.asksaveasfilename(defaultextension='.json',filetypes=[('JSON','*.json')])
        if p: self.form_model_path.set(p)
    def flog(self,s=''):
        self.formlog.insert(tk.END,str(s)+'\n'); self.formlog.see(tk.END); self.log(s)
    def fllogmsg(self,s=''):
        self.fllog.insert(tk.END,str(s)+'\n'); self.fllog.see(tk.END); self.log(s)
    def sammsg(self,s=''):
        try: self.samlog.insert(tk.END,str(s)+'\n'); self.samlog.see(tk.END)
        except Exception: pass
        self.log(s)
    def train_formlearner(self):
        try:
            root=Path(self.form_project.get() or self.project.get()).expanduser(); imgs_dir=root/'images'; labs_dir=root/'labels'
            if not imgs_dir.exists(): raise RuntimeError('Projektordner braucht images/')
            if not labs_dir.exists(): raise RuntimeError('Projektordner braucht labels/')
            X=[]; y=[]; nimg=0
            for img in sorted([p for p in imgs_dir.iterdir() if p.suffix.lower() in IMG_EXT]):
                lab=labs_dir/(img.stem+'.txt')
                if not lab.exists(): continue
                im=Image.open(img).convert('RGB'); W,H=im.size; nimg+=1
                for line in lab.read_text(encoding='utf-8',errors='ignore').splitlines():
                    p=line.split()
                    if len(p)!=5: continue
                    cls=int(float(p[0])); cx,cy,bw,bh=map(float,p[1:])
                    x1=(cx-bw/2)*W; y1=(cy-bh/2)*H; x2=(cx+bw/2)*W; y2=(cy+bh/2)*H
                    X.append(crop_features(im,(x1,y1,x2,y2))); y.append(1 if cls==0 else 0)
            pos=sum(y); neg=len(y)-pos
            self.flog(f'FormTrainer Samples: {len(y)} | positiv={pos} | false={neg} | Bilder={nimg}')
            if len(set(y))<2: raise RuntimeError('Für FormTrainer werden positive UND false-positive Labels benötigt.')
            model=SimpleFormLearner(); model.fit(X,y,epochs=int(self.form_epochs.get()))
            out=Path(self.form_model_path.get() or str(root/'formlearner_model.json')); out.parent.mkdir(parents=True,exist_ok=True); model.save(out)
            self.fl_model_path.set(str(out)); self.flog(f'FormLearner gespeichert: {out}')
        except Exception as e:
            self.flog('FORMTRAINER ERROR '+str(e)); messagebox.showerror(APP,str(e))
    def draw_fl_preview(self):
        try:
            if not hasattr(self,'fl_canvas') or not getattr(self,'fl_preview_img',None): return
            img,OW,OH=load_preview(self.fl_preview_img,1800)
            cw=max(10,self.fl_canvas.winfo_width()); ch=max(10,self.fl_canvas.winfo_height())
            sc=min(cw/img.width,ch/img.height); nw=max(1,int(img.width*sc)); nh=max(1,int(img.height*sc))
            im=img.resize((nw,nh)); self.fl_photo=ImageTk.PhotoImage(im)
            self.fl_canvas.delete('all'); x0=(cw-nw)//2; y0=(ch-nh)//2
            self.fl_canvas.create_image(x0,y0,anchor='nw',image=self.fl_photo)
            sx=nw/OW; sy=nh/OH
            colors=['lime','cyan','magenta','yellow','orange','red']
            for rank,(d,fs) in enumerate(getattr(self,'fl_kept',[]),1):
                if fs < float(self.fl_threshold.get()): continue
                x1,y1,x2,y2=d.bbox(); col=colors[getattr(d,'slot',0)%len(colors)]
                self.fl_canvas.create_rectangle(x0+x1*sx,y0+y1*sy,x0+x2*sx,y0+y2*sy,outline=col,width=2)
                self.fl_canvas.create_text(x0+x1*sx+3,y0+y1*sy+3,anchor='nw',fill=col,text=f'{rank} M{getattr(d,"slot",0)+1} F{fs:.2f} C{d.conf:.2f}')
        except Exception as e:
            try: self.fllogmsg('FL Vorschau Fehler: '+str(e))
            except Exception: pass

    def detect_with_formlearner(self):
        try:
            model=SimpleFormLearner.load(self.fl_model_path.get())
            self.fllogmsg('Starte Basis-Detection...')
            self.detect()
            img=Path(getattr(self,'last_img',Path(self.image.get()))); W,H,mode,reader=open_img(img,self.fllogmsg)
            kept=[]; threshold=float(self.fl_threshold.get())
            for i,d in enumerate(self.dets,1):
                if d.score < float(self.minscore.get()): continue
                x1,y1,x2,y2=d.bbox(); pad=32; left=max(0,int(x1-pad)); top=max(0,int(y1-pad)); side=max(32,int(max(x2-x1,y2-y1)+2*pad)); crop=read_tile(reader,mode,left,top,side,W,H)
                score=model.predict(crop_features(crop))
                if score>=threshold:
                    kept.append((d,score))
                if i%25==0: self.fllogmsg(f'FormFilter Stand {i}/{len(self.dets)} | behalten={len(kept)}')
            if mode=='rasterio': reader.close()
            feats=[]
            for rank,(d,fs) in enumerate(kept,1):
                x1,y1,x2,y2=d.bbox(); feats.append({'type':'Feature','geometry':{'type':'Polygon','coordinates':[[(x1,y1),(x2,y1),(x2,y2),(x1,y2),(x1,y1)]]},'properties':{'rank':rank,'model':d.name,'confidence':d.conf,'ensemble_score':d.score,'form_score':fs,'pixel_bbox':[x1,y1,x2,y2]}})
            out=Path(self.fl_output.get() or str(img.with_suffix('.formlearner.geojson')))
            if out.suffix.lower()=='.gpkg':
                try:
                    write_gpkg_fallback(out, feats, layer='formlearner_detections', crs_name=None, log_fn=self.fllogmsg)
                except Exception as e:
                    self.fllogmsg('GPKG fehlgeschlagen, schreibe GeoJSON: '+str(e)); out=out.with_suffix('.geojson'); out.write_text(json.dumps({'type':'FeatureCollection','features':feats},indent=2),encoding='utf-8')
            else:
                out.write_text(json.dumps({'type':'FeatureCollection','features':feats},indent=2),encoding='utf-8')
            self.fl_kept=kept
            self.fl_preview_img=img
            self.draw_fl_preview()
            self.fllogmsg(f'Detection mit FormLearner fertig: {len(kept)} Treffer -> {out}')
        except Exception as e:
            self.fllogmsg('FORMLEARNER DETECTION ERROR '+str(e)); messagebox.showerror(APP,str(e))

if __name__=='__main__': GUI().mainloop()
