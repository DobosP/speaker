# Free Spoken Digit Dataset adapted fixtures

The arrays under `real_usage_full/` and `virtual_real_world/` are adapted from
recordings in the
[Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset).
The project's credited creators are Zohar Jackson, César Souza, Jason Flaks,
Yuxin Pan, Hereman Nicolas, and Adhish Thite. The source speaker identifiers
retained in metadata are `george`, `jackson`, `lucas`, `nicolas`, and `theo`.

The original recordings are licensed under the
[Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/).
The adapted fixture arrays are distributed under the same CC BY-SA 4.0
license. They are not covered by the repository's MIT software license.

Changes made by the speaker project include resampling to 16 kHz and applying
deterministic gain, distance, room, noise, playback-device, clipping, and
sample-rate transformations. Exact case-level source filenames and generated
signal properties are recorded in each directory's `metadata.json`.

The historical generator did not record the exact upstream FSDD tag or commit.
That revision is therefore reported as unknown; no revision has been inferred
or invented.
