## Setup środowiska
* Instalujemy conda'ę (można lokalnie tylko dla własnego użytkownika), najlepiej z [conda-forge](https://conda-forge.org/download/)
* Instalujemy środowisko `CONDA_OVERRIDE_CUDA=12.2 conda env create -f environment.yml -p ./condaenv`
* Aktywujemy `conda activate ścieżka/do/condaenv`.

## Trenowanie
* Jeśli mamy wiele kart do wyboru (`nvidia-smi` je pokaże) to wybieramy np. jedną dopisując przed komendą `CUDA_VISIBLE_DEVICES=0 python ...`.
* Transfer learning: modyfikujemy `lightning_configs/classifier.yaml` i uruchamiamy `python -m lightning_modules.classifier fit --config lightning_configs/classifier.yaml`
* Oglądanie wykresów: `tensorboard --logdir=checkpoints/ --port=6056`

## SLURM
* Uruchomienie jobu: `JOB_ID=$(sbatch job-classifier.sh | awk '{print $NF}')` (ustawiamy w `job-classifier.sh` time limit, konfigurację w `classifier.yaml`).
* Oglądanie outputu na bieżąco: `tail --follow checkpoints/*/s$JOB_ID/stdout`
* Oglądanie statusu GPU: `srun --jobid $JOB_ID nvidia-smi` (działa tylko jeśli mamy dostęp do dodatkowych zasobów w kolejce).
* Ubijanie jobu: `scancel --me --signal=SIGINT $JOB_ID`
* Anulowanie czekających na alokację job'ów `scancel --me -t PENDING`
* Działające joby: `squeue --me`
* Zakończone joby: `sacct  --format=JobID,Start,End,Elapsed,NCPUS,NodeList,NTasks,ExitCode,JobName,User,State`
* Wznawianie treningu z checkpointu: patrz `job-continue.sh`

## saved_models/
Contains checkpoints with model weights that we want to keep:
- `downloaded/` downloaded from https://github.com/yitu-opensource/T2T-ViT
- `classifier/` trenowane z `python -m lightning_modules.classifier fit --config .../config.yaml`
    - `cifar10/`
        - `v4/`
            - `vit_small_patch16_224`        val accuracy 97.7% without masks. (powinno dać się 98.2%)
            - `swin_tiny_patch4_window7_224` val accuracy 98.0% without masks.
            - `t2t_vit_14`                   val accuracy TODO without masks.
    - `gastro/`
        - `v4/`
            - `vit_small_patch16_224`        val accuracy 96.3% without masks. (pewnie da się lepiej)
            - `swin_tiny_patch4_window7_224` val accuracy 98.5% without masks. (powinno dać się 99.2%)
            - `t2t_vit_14`                   TODO
- `surrogate`  trenowane z `python -m lightning_modules.surrogate fit --config .../config.yaml`
    -  `cirfar10/`
        - `v4/`
            - `player16/`
                - `vit_small_patch16_224`         val accuracy unmasked 98.2%, 16-masked 85.9%
                - `swin_tiny_patch4_window7_224`  val accuracy unmasked 98.1%, 16-masked 86.3%
                - `t2t_vit_14`                    TODO
            - `player196/`
                - `vit_small_patch16_224`         val accuracy unmasked 97.9%, 196-masked 92.6%
                - `swin_tiny_patch4_window7_224`  val accuracy unmasked 98.0%, 196-masked 92.8%
                - `t2t_vit_14`                    TODO
    - `gastro/`
        - `v4/`
            - `player16/`
                - `vit_small_patch16_224`         val accuracy unmasked 97.2%, 16-masked 90.8%
                - `swin_tiny_patch4_window7_224`  val accuracy unmasked 99.7%, 16-masked 96.1%
                - `t2t_vit_14`                    TODO
            - `player196/`
                - `vit_small_patch16_224`         val accuracy unmasked 96.0%, 196-masked 93.7%
                - `swin_tiny_patch4_window7_224`  val accuracy unmasked 99.0%, 196-masked 98.3%
                - `t2t_vit_14`                    TODO
- `explainer` trenowane z `python -m lightning_modules.explainer fit --config .../config.yaml`
  -  `cirfar10/`
        - `v4/`
            - `player16/`
                - `vit_small_patch16_224`         val-macc-best 65.9%
                - `swin_tiny_patch4_window7_224`  val-macc-best 65.0%
                - `t2t_vit_14`                    TODO
            - `player196/`
                - `vit_small_patch16_224`         val-macc-best TODO
                - `swin_tiny_patch4_window7_224`  val-macc-best 61.1%
                - `t2t_vit_14`                    TODO
    - `gastro/`
        - `v4/`
            - `player16/`
                - `vit_small_patch16_224`         val-macc-best TODO
                - `swin_tiny_patch4_window7_224`  TODO
                - `t2t_vit_14`                    TODO
            - `player196/`
                - `vit_small_patch16_224`         TODO
                - `swin_tiny_patch4_window7_224`  TODO
                - `t2t_vit_14`                    TODO



### Przed refactorem v4:

- `explainer/cifar10/`:
    - `v2/`:
        - `t2t_vit.ckpt`: trained for 402 epochs (17k steps).
    - `v3/`:
        trained with `CUDA_VISIBLE_DEVICES=x python -m vit_shapley.modules.explainer --num_players NNN --target_model_name=X --backbone_name=X --lr 0.00005 --acc 4 --divisor=$((NNN/2))`
        (some with --use_sb=true)
        - `player196/t2t_vit.ckpt`: 306 epochs, val/macc-best ~50%
        - `player196/swin.ckpt`:    499 epochs, val/macc-best ~70% (could train for more)
        - `player196/vit.ckpt`:     211 epochs, val/macc-best ~55%
        - `player16/t2t_vit.ckpt`: 198 epochs, val/macc-best ~68% (100 epochs would have been fine)
        - `player16/swin.ckpt`:    499 epochs, val/macc-best ~78%
        - `player16/vit.ckpt`:     106 epochs, val/macc-best ~70%
