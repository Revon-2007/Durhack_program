# smart-nappy-ai

Reads a photo of a microfluidic nappy indicator patch and estimates liquid volume two independent ways.

    photo -> white balance (reference patch)
          -> A) chamber colours  -> trained model -> pH + colour-based volume
          -> B) channel fronts   -> filled length (mm) -> V = width x depth x length
                                 -> calibration curve -> total nappy volume
          -> cross-check A vs B, warnings, overlay image

## Quick start
```
pip install -r requirements.txt
python scripts/make_synthetic_data.py --n 200       # SYNTHETIC data, for testing only
python -m src.training.train_colour                 # colour -> pH / volume models
python -m src.training.calibrate_volume             # channel length -> nappy volume curve
python -m src.inference.predict_image data/raw/images/synthetic_0003.png
pytest
```
Outputs go to `outputs/predictions/`: a JSON result and `*_overlay.png`
(green box = channel, red line = detected liquid front) so you can verify what it saw.

## Channel detection
Per channel: Lab colour projected onto the dry->wet axis, median across the channel width,
smoothed, thresholded at 0.5, small bubbles bridged (`gap_tol_frac`), front refined to sub-pixel.
Channels are horizontal with the inlet on the LEFT; each ROI must span the full physical
`length_mm`. Edit positions/lengths under `channels.items` in `config.yaml`.

## Using real data
1. Print the patch with a white reference square; match `layout` and `channels.items` to it.
2. Photograph a DRY patch and a fully WET one; set `channels.dry_rgb` / `wet_rgb`
   (white-balanced colours: run extract on those photos and read the channel colour).
3. Photograph at known volumes. Fill `data/raw/measurements.csv`:
   `image,batch,ph,volume_ml` and optionally `ch1_len_mm,ch2_len_mm,...` (ruler-measured true
   fronts - lets calibrate_volume report detection error in mm). One `batch` per run/sample.
4. Retrain and calibrate; check `models/trained/metrics.json` and `calibration_metrics.json`.

## Honest limits
- Synthetic images use idealised colours and a clean meniscus; real channels will be messier
  (uneven wetting, stained paper, glare). Expect the real error to be much larger.
- V = A x L is only the volume INSIDE the channels. Total nappy volume comes from the fitted
  calibration curve, which is only valid inside the range you calibrated.
- A pH dye cannot report ammonia; that needs its own assay.
