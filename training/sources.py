"""Which Quick, Draw! categories teach the guesser each game word.

Most words use their own category. Some things are commonly drawn in another
form that Quick, Draw! keeps as a separate category (a ceiling fan instead of
a desk fan), so those drawings are mixed into the word's training data. Without
this, a player who draws a ceiling fan for "fan" gets "dolphin" or
"windmill" back.
"""

# word -> [(Quick, Draw! category, share of the word's training drawings)]
VARIANTS = {
    'alarm_clock': [('alarm clock', 0.7), ('clock', 0.3)],
    'bus': [('bus', 0.7), ('school bus', 0.3)],
    'face': [('face', 0.6), ('smiley face', 0.4)],
    'fan': [('fan', 0.6), ('ceiling fan', 0.4)],
    'truck': [('truck', 0.7), ('pickup truck', 0.3)],
}


def sources_for(word):
  return VARIANTS.get(word, [(word.replace('_', ' '), 1.0)])


def all_categories(words):
  return sorted({category for w in words for category, _ in sources_for(w)})


def raw_name(category):
  return category.replace(' ', '_') + '.json'
