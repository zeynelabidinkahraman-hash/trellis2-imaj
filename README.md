# trellis2-imaj — TRELLIS.2 hazır üretim imajı

**Neden var:** 3 haftada 5 kiralık pod kurulumu ayrı sebeplerle çöktü (conda ToS/Python 3.14, torch 2.14+cu130,
flash-attn kaynak derlemesi, transformers 5.x). Her podda sıfırdan kurulum ~50–85 dk ve ~$0,55 yiyordu.
Kurulum bir kez burada yapılır, kapıdan geçer, dondurulur.

**İmaj:** `ghcr.io/zeynelabidinkahraman-hash/trellis2:cu124-torch26`
- CUDA 12.4 devel · Python 3.10 venv (/workspace/venv) · torch 2.6.0+cu124 · flash-attn 2.7.3 (hazır wheel)
- nvdiffrast 0.4.0, nvdiffrec, CuMesh, FlexGEMM, o_voxel — sm 8.6 (3090) + 8.9 (4090) için derli
- transformers 4.57.1 · TRELLIS.2 commit 75fbf018 · **microsoft/TRELLIS.2-4B ağırlıkları gömülü** (HF_HOME=/workspace/hf)
- DINOv3 gated → gömülü DEĞİL; pod'a `HF_TOKEN` env verilir, ilk çalıştırmada ~1,2 GB iner

**vast.ai kullanımı** (arac/vast-kirala.js `--imaj` ile):
1. Pod aç: image = yukarıdaki etiket, env `-e HF_TOKEN=...`, disk ≥ 40 GB
2. `scp girdi/*.png root@pod:/workspace/girdi/`
3. `ssh root@pod bash /workspace/t2-basla.sh` → /workspace/uretim.log, çıktılar /workspace/cikti-toplu/*.glb
4. Bekçi: safir-pulse-cat-katalog/arac/t2-uretim-bekci-1909.sh (indir + imha)

**Beklenen süre/maliyet (22 parça):** pod açılış ~3 dk + DINOv3 ~5 dk + 22 × ~100 sn ≈ 45 dk; 3090 ≈ $0,15, 4090 ≈ $0,30.

**İnşa:** `.github/workflows/imaj.yml` — push ile veya elle (Actions → Run workflow). İlk inşa ~60–90 dk (uzantı derlemesi + 8 GB ağırlık), sonrakiler katman önbelleğiyle dakikalar.
