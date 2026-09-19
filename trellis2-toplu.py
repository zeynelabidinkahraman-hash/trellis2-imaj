#!/usr/bin/env python3
"""trellis2-toplu.py — TRELLIS.2 (4B) toplu üretim; pod üzerinde koşar.

Kesinti güvenli: var olan GLB atlanır, 2 kez düşen girdi işaretlenip bir daha denenmez.
31.08 dersleri gömülü:
  · Beyaz zeminli çizim PNG'leri saydama çevrilir (RMBG hiç çağrılmaz — o model de kapılı).
  · İlerleme satırları FLUSH'lı ve \n'li (eski log \r yüzünden sayılamıyordu).
  · OOM/ağır remesh gerçek → dosya başına try/except, GPU önbelleği her turda boşaltılır.
Kullanım: python trellis2-toplu.py --girdi=/workspace/girdi --cikti=/workspace/cikti-t2
"""
import os, sys, gc, time, argparse, traceback
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
from PIL import Image
import torch

# ⚠ RMBG TUZAĞI — 19.09'da BU BETİKTE tekrar yaşandı (t2-dene.py'de yama vardı, burada YOKTU;
# üstteki yorum "RMBG hiç çağrılmaz" diyordu ama fiilen ENGELLENMİYORDU).
# TRELLIS arka plan silmek için briaai/RMBG-2.0 kullanıyor, repo GATED (403).
# Hata run() sırasında DEĞİL, from_pretrained() SIRASINDA oluşur:
#   pipeline.rembg_model = getattr(rembg, args['rembg_model']['name'])(...)
# Bu yüzden "girdiyi RGBA ver" tek başına yetmez — oraya hiç gelinmez. Sınıf, pipeline
# kurulmadan ÖNCE sahteyle değiştirilir. Alpha maskesini beyazdan_alfa() zaten üretiyor;
# yine de çağrılırsa SESSİZ GEÇMEZ, patlar.
import trellis2.pipelines.rembg as _rembg_mod
class _RembgKapali:
    def __init__(self, *a, **k): pass
    def to(self, *a, **k): return self
    def cpu(self, *a, **k): return self
    def __call__(self, *a, **k):
        raise RuntimeError('rembg kapali: girdi RGBA olmali (alpha maskesi bizde uretiliyor)')
_rembg_mod.BiRefNet = _RembgKapali

from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel
import transformers

ap = argparse.ArgumentParser()
ap.add_argument('--girdi', required=True)
ap.add_argument('--cikti', required=True)
ap.add_argument('--isaret', default='/workspace/isaret')
ap.add_argument('--doku', type=int, default=2048)      # 4096 katalog için gereksiz büyük
ap.add_argument('--yuz', type=int, default=300000)     # decimation hedefi
a = ap.parse_args()
os.makedirs(a.cikti, exist_ok=True); os.makedirs(a.isaret, exist_ok=True)
IMZA = 'tf' + transformers.__version__ + '-v2'   # ortam imzasi: degisince eski dusus isaretleri gecersiz

def beyazdan_alfa(p):
    """Cizim PNG'leri beyaz zeminli; TRELLIS alfa bekler. Beyazi saydam yap.
    11.09 dersi: sabit esik (>240) acik-gri cizimlerde HER SEYI saydam yapiyor -> preprocess_image'in
    bbox'i bos kume ('zero-size array to reduction'). Esik KADEMELI yukseltilir; hicbiri yeterli opak
    piksel birakmazsa girdi TAM OPAK gonderilir (sessizce bos GLB uretmek yerine)."""
    im = Image.open(p).convert('RGBA')
    d = np.array(im)
    toplam = d.shape[0] * d.shape[1]
    for esik in (240, 248, 252, 254):
        beyaz = (d[..., 0] > esik) & (d[..., 1] > esik) & (d[..., 2] > esik)
        opak = toplam - int(beyaz.sum())
        if opak >= toplam * 0.001:          # en az binde bir piksel nesne olmali
            d[..., 3] = np.where(beyaz, 0, 255)
            return Image.fromarray(d), esik
    d[..., 3] = 255
    return Image.fromarray(d), None

def isaret_oku(yol, imza):
    """Isaret = '<deneme>|<ortam imzasi>'. Ortam degistiyse eski dususler GECERSIZ sayilir.
    11.09 dersi: transformers yanlis surumdeyken 375 girdinin hepsi isaretlenmisti; ortam duzelince
    imza degistigi icin sayac sifirlanir, yoksa tur sessizce 'hepsi atlandi' derdi."""
    try:
        ham = open(yol).read().strip()
        parca = (ham.split('|') + [''])[:2]
        return int(parca[0]) if parca[1] == imza else 0
    except Exception:
        return 0

print('[t2] boru hattı yükleniyor...', flush=True)
pipe = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
pipe.cuda()
print('[t2] hazır', flush=True)

girdiler = sorted(f for f in os.listdir(a.girdi) if f.lower().endswith('.png'))
ok = atla = hata = 0
for i, f in enumerate(girdiler, 1):
    ad = os.path.splitext(f)[0]
    hedef = os.path.join(a.cikti, ad + '.glb')
    isaret = os.path.join(a.isaret, ad + '.bad')
    if os.path.exists(hedef): atla += 1; continue
    if isaret_oku(isaret, IMZA) >= 2: atla += 1; continue
    t0 = time.time()
    try:
        img, esik = beyazdan_alfa(os.path.join(a.girdi, f))
        mesh = pipe.run(img)[0]
        mesh.simplify(16777216)                       # nvdiffrast sınırı
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
            coords=mesh.coords, attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=a.yuz, texture_size=a.doku,
            remesh=True, remesh_band=1, remesh_project=0, verbose=False)
        gecici = hedef[:-4] + '.part.glb'   # uzanti .glb KALMALI: trimesh bicimi uzantidan secer
        glb.export(gecici, extension_webp=True)
        os.replace(gecici, hedef)             # yarım dosya kataloğa sızmasın
        ok += 1
        print('[%d/%d] ✓ %s %.0fsn %.1fMB' % (i, len(girdiler), ad, time.time() - t0,
              os.path.getsize(hedef) / 1048576), 'esik=' + str(esik), flush=True)
    except Exception as e:
        hata += 1
        n = isaret_oku(isaret, IMZA)
        open(isaret, 'w').write(str(n + 1) + '|' + IMZA)
        print('[%d/%d] ✗ %s (%d. deneme) %s' % (i, len(girdiler), ad, n + 1,
              str(e)[:120]), flush=True)
        traceback.print_exc(limit=2)
    finally:
        gc.collect(); torch.cuda.empty_cache()
print('BİTTİ · üretilen=%d atlanan=%d hata=%d' % (ok, atla, hata), flush=True)
