// Word display and guess checking. The word list itself comes from the
// classifier (SKETCH_CLASSIFIER_WEIGHTS.words), so both halves of the game
// always use the same words.
(function (root) {
  'use strict';

  // Other names players commonly type for a word.
  const ALIASES = {
    alarm_clock: ['clock'],
    backpack: ['bag', 'rucksack', 'schoolbag'],
    barn: ['farmhouse'],
    bicycle: ['bike', 'cycle'],
    bulldozer: ['dozer', 'digger'],
    cat: ['kitten', 'kitty'],
    couch: ['sofa'],
    cruise_ship: ['ship', 'cruise'],
    diving_board: ['diving'],
    dog: ['puppy', 'doggy'],
    face: ['smiley', 'smiley face', 'head'],
    fire_hydrant: ['hydrant'],
    firetruck: ['fire engine'],
    helicopter: ['chopper'],
    lantern: ['lamp'],
    paintbrush: ['brush'],
    palm_tree: ['palm', 'tree'],
    peas: ['pea', 'pea pod', 'peapod'],
    pool: ['swimming pool'],
    postcard: ['card'],
    power_outlet: ['outlet', 'socket', 'plug', 'power socket'],
    rabbit: ['bunny'],
    rain: ['rain cloud', 'cloud'],
    rhinoceros: ['rhino'],
    rifle: ['gun'],
    roller_coaster: ['coaster'],
    sea_turtle: ['turtle'],
    speedboat: ['boat', 'motorboat'],
    stove: ['oven', 'cooker'],
    swing_set: ['swing', 'swings'],
    truck: ['lorry'],
    yoga: ['yoga pose'],
  };

  // Words that read wrong with "a"/"an" in front of them.
  const NO_ARTICLE = new Set(['rain', 'peas', 'toothpaste', 'yoga']);

  function display(word) {
    return word.replace(/_/g, ' ');
  }

  function withArticle(word) {
    const text = display(word);
    if (NO_ARTICLE.has(word)) return text;
    return (/^[aeiou]/.test(text) ? 'an ' : 'a ') + text;
  }

  function normalize(text) {
    return text.toLowerCase().replace(/[^a-z]/g, '');
  }

  function editDistance(a, b) {
    const row = Array.from({ length: b.length + 1 }, (_, i) => i);
    for (let i = 1; i <= a.length; i++) {
      let prev = row[0];
      row[0] = i;
      for (let j = 1; j <= b.length; j++) {
        const tmp = row[j];
        row[j] = Math.min(row[j] + 1, row[j - 1] + 1,
                          prev + (a[i - 1] === b[j - 1] ? 0 : 1));
        prev = tmp;
      }
    }
    return row[b.length];
  }

  // Returns 'correct', 'close' (a typo away) or 'wrong'.
  function checkGuess(guess, word) {
    const g = normalize(guess);
    if (!g) return 'wrong';
    const accepted = [word, ...(ALIASES[word] || [])].map(normalize);
    const plural = (w) => [w, w + 's', w + 'es'];
    if (accepted.some((a) => plural(a).includes(g))) return 'correct';
    const target = normalize(word);
    const allowed = target.length >= 7 ? 2 : 1;
    return editDistance(g, target) <= allowed ? 'close' : 'wrong';
  }

  root.Words = { display, withArticle, checkGuess };
})(window);
