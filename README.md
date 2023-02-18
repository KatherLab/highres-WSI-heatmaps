# High-Resolution Heatmaps for Marugoto MIL Models

## Options

```sh
create_heatmaps.py [-h] -m MODEL_PATH -o OUTPUT_PATH (-t TRUE_CLASS | --is-regression)
                   [--no-pool]
                   [--mask-threshold THRESH]
                   [--att-upper-threshold THRESH]
                   [--att-lower-threshold THRESH]
                   [--score-threshold THRESH]
                   [--att-cmap CMAP]
                   [--score-cmap CMAP]
                   SLIDE [SLIDE ...]
```

Create heatmaps for classification or regression attMIL models.

| Positional Arguments | Description |
|----------------------|-------------|
| `SLIDE` | Slides to create heatmaps for.  If multiple slides are given, the normalization of the attention / score maps' intensities will be performed across all slides. |

| Options | Description |
|---------|-------------|
| `-m MODEL_PATH`, `--model-path MODEL_PATH` | MIL model used to generate attention / score maps. |
| `-o OUTPUT_PATH`, `--output-path OUTPUT_PATH` | Path to save results to. |
| `--no-pool` | Do not average pool features after feature extraction phase. |
| `--cache-dir CACHE_DIR` | Directory to cache extracted features etc. in. |

| Model type (mutually exclusive) | Description |
|---------|-------------|
| `-t TRUE_CLASS`, `--true-class TRUE_CLASS` | Using a classification model for heatmaps, class to be rendered as "hot" in the heatmap for classification. |
| `--use-regression` | Using a regression model for heatmaps, doesn't work for score maps yet. |

| Thresholds | Description |
|------------|-------------|
| `--mask-threshold THRESH` | Brightness threshold for background removal. |
| `--att-upper-threshold THRESH` | Quantile to squash attention from during attention scaling (e.g. 0.99 will lead to the top 1% of attention scores to become 1) |
| `--att-lower-threshold THRESH` | Quantile to squash attention to during attention scaling (e.g. 0.01 will lead to the bottom 1% of attention scores to become 0) |
| `--score-threshold THRESH` | Quantile to consider in score scaling (e.g. 0.95 will discard the top / bottom 5% of score values as outliers) |

| Colors | Description |
|--------|-------------|
| `--att-cmap CMAP` | Color map to use for the attention heatmap. |
| `--score-cmap CMAP` | Color map to use for the score heatmap. |

## Running in a Container

The heatmap script can be conveniently run in a podman container.  To do so, use
the `heatmaps-container.sh` convenience script.

```sh
./heatmaps-container.sh \
    -t TRUE_CLASS \
    /wsis/slide1.svs \
    /wsis/slide2.svs
```

In order to use GPU acceleration, the `nvidia-container-toolkit` has to be
installed beforehand.

