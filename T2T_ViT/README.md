## Setup środowiska
* Instalujemy conda'ę (można lokalnie tylko dla własnego użytkownika), najlepiej z [conda-forge](https://conda-forge.org/download/)
* Instalujemy środowisko `CONDA_OVERRIDE_CUDA=12.2 conda env create -f environment.yml -p ./condaenv`
* Aktywujemy `conda activate ścieżka/do/condaenv`.

## Trenowanie
* Jeśli mamy wiele kart do wyboru (`nvidia-smi` je pokaże) to wybieramy np. jedną dopisując przed komendą `CUDA_VISIBLE_DEVICES=0 python ...`.
* Transfer learning: modyfikujemy `lightning_configs/classifier.yaml` i uruchamiamy `python -m lightning_modules.classifier fit --config lightning_configs/classifier.yaml`
    Można też dopisywać opcje do linii komend po `--config ...yaml`, np. `--data.dataloader_kwargs.batch_size=256 --trainer.max_epochs=40`.
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
            - `vit_small_patch16_224`        val accuracy 97.7%. (powinno dać się 98.2%)
            - `swin_tiny_patch4_window7_224` val accuracy 98.0%.
            - `t2t_vit_14`                   val accuracy 97.7%.
    - `gastro/`
        - `v4/`
            - `vit_small_patch16_224`        val accuracy 96.3%. (pewnie da się lepiej)
            - `swin_tiny_patch4_window7_224` val accuracy 98.5%. (powinno dać się 99.2%)
            - `t2t_vit_14`                   val accuracy 99.0%.
        - `v5/` uczone na gastro z "cropped"
            - `vit_small_patch16_224`        val accuracy 90.5%.
            - `swin_tiny_patch4_window7_224` val accuracy 91.3%.
            - `t2t_vit_14`                   val accuracy 93.7%.
- `surrogate`  trenowane z `python -m lightning_modules.surrogate fit --config .../config.yaml`
    -  `cifar10/`
        - `v4/`
            - `player16/`
                - `vit_small_patch16_224`         val accuracy unmasked 98.2%, 16-masked 85.9%
                - `swin_tiny_patch4_window7_224`  val accuracy unmasked 98.1%, 16-masked 86.3%
                - `t2t_vit_14`                    val accuracy unmasked 98.0%, 16-masked 85.5%
            - `player196/`
                - `vit_small_patch16_224`         val accuracy unmasked 97.9%, 196-masked 92.6%
                - `swin_tiny_patch4_window7_224`  val accuracy unmasked 98.0%, 196-masked 92.8%
                - `t2t_vit_14`                    val accuracy unmasked 97.7%, 196-masked 91.5%
    - `gastro/`
        - `v4/`
            - `player16/`
                - `vit_small_patch16_224`         val accuracy unmasked 97.2%, 16-masked 90.8%
                - `swin_tiny_patch4_window7_224`  val accuracy unmasked 99.7%, 16-masked 96.1%
                - `t2t_vit_14`                    val accuracy unmasked 98.2%, 16-masked 93.3%
            - `player196/`
                - `vit_small_patch16_224`         val accuracy unmasked 96.0%, 196-masked 93.7%
                - `swin_tiny_patch4_window7_224`  val accuracy unmasked 99.0%, 196-masked 98.3%
                - `t2t_vit_14`                    val accuracy unmasked 98.7%, 196-masked 94.9%
        - `v5/` uczone na gastro z "cropped"
            - `player16/`
                - `vit_small_patch16_224`         val accuracy unmasked 85.7%, 16-masked 92.7%
                - `swin_tiny_patch4_window7_224`  val accuracy unmasked 93.2%, 16-masked 88.2%
                - `t2t_vit_14`                    val accuracy unmasked 92.7%, 16-masked 88.0%
            - `player196/`
                - `vit_small_patch16_224`         val accuracy unmasked 92.5%, 196-masked 89.2%
                - `swin_tiny_patch4_window7_224`  val accuracy unmasked 91.8%, 196-masked 89.1%
                - `t2t_vit_14`                    val accuracy unmasked 93.7%, 196-masked 89.8%
- `explainer` trenowane z `python -m lightning_modules.explainer fit --config .../config.yaml`
  -  `cifar10/`
        - `v4/`
            - `player16/`
                - `vit_small_patch16_224`         val-macc-best 65.9%,
                - `swin_tiny_patch4_window7_224`  val-macc-best 65.0%,
                - `t2t_vit_14`                    val-macc-best 61.9%
            - `player196/`
                - `vit_small_patch16_224`         val-macc-best 55.4%,
                - `swin_tiny_patch4_window7_224`  val-macc-best 63.1%,
                - `t2t_vit_14`                    val-macc-best 45.2%
    - `gastro/`
        - `v4/`
            - `player16/`
                - `vit_small_patch16_224`         val-macc-best 72.3%,
                - `swin_tiny_patch4_window7_224`  val-macc-best 87.3% (perhaps because surrogate achieves 99.7%)
                - `t2t_vit_14`                    val-macc-best 66.9%
            - `player196/`
                - `vit_small_patch16_224`         val-macc-best 27.4%
                - `swin_tiny_patch4_window7_224`  val-macc-best 64.3%
                - `t2t_vit_14`                    val-macc-best 25.8%
        - `v5/`
            - `player196/`
                - `vit_small_patch16_224`         val-macc-best 27.8%
                - `swin_tiny_patch4_window7_224`  val-macc-best 44.2%
                - `t2t_vit_14`                    val-macc-best 26.8%