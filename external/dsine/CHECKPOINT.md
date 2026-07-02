# DSINE checkpoint (not tracked in git)

This project needs the pretrained DSINE checkpoint (Bae & Davison, CVPR 2024)
to run. It is not committed to this repo: it's 278MB, over GitHub's 100MB
push limit, and it's a third-party artifact anyway (this repo only contains
the code that loads it).

## How to get it

1. Go to the official checkpoints folder (linked from
   https://github.com/baegwangbin/DSINE, maintained by the paper authors):
   https://drive.google.com/drive/folders/1t3LMJIIrSnCGwOEf53Cyg0lkSXd3M4Hm

2. Open `checkpoints/exp001_cvpr2024/` and download only `dsine.pt`
   (this is the "best model" from the paper; ignore `exp002_kappa/`, a
   variant that also estimates uncertainty -- not used here).

3. Save it to:
   `external/dsine/checkpoints/exp001_cvpr2024/dsine.pt`

Once it's in place, `main.py` and the `test_*.py` scripts will find it
automatically (path is hardcoded to this location).
