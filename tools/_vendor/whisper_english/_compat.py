"""The full three-word windows required by the pinned English normalizer.

EnglishNumberNormalizer always supplies [None] + nonempty_words + [None],
with n=3. This intentionally does not implement more_itertools' unrelated
padding/step/general-window API. No new text normalization rules are added.
"""


def windowed(words, n):
    if n != 3 or type(words) is not list or len(words) < 3:
        raise ValueError("unsupported_normalizer_window")
    previous, current = words[0], words[1]
    for following in words[2:]:
        yield previous, current, following
        previous, current = current, following
