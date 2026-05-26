# Third-party datasets used for testing

The sample slices under `samples/` and the full datasets fetched into `_cache/`
(when `SPEAKER_DATASET_DOWNLOAD=1`) come from these open-source corpora:

## CLINC150
- Source: https://github.com/clinc/oos-eval
- Paper: Larson et al., "An Evaluation Dataset for Intent Classification and
  Out-of-Scope Prediction" (EMNLP 2019).
- License: Creative Commons Attribution 3.0 (CC BY 3.0).

## MASSIVE 1.0
- Source: https://github.com/alexa/massive
- Paper: FitzGerald et al., "MASSIVE: A 1M-Example Multilingual Natural Language
  Understanding Dataset with 51 Typologically-Diverse Languages" (2022).
- License: Creative Commons Attribution 4.0 (CC BY 4.0).

The committed `samples/*.json` files contain small slices of the above, retained
under the same licenses for offline/CI testing. The full datasets are not
committed; they are downloaded on demand and cached locally (gitignored).
