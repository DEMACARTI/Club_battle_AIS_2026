// Game flow: alternates "AI draws, you guess" and "you draw, AI guesses"
// rounds, keeps score, and shows the results gallery.
(function () {
  'use strict';

  const ROUNDS = 6;
  const DIFFICULTY = {
    // temperature: how loosely the AI draws. candidates: how many drawings
    // it samples before showing the one the classifier finds clearest.
    easy: {
      temperature: 0.1, candidates: 6, guessSeconds: 45, drawSeconds: 30,
      hintAt: [0, 0.35, 0.65],
      note: 'The AI draws its clearest sketches, letter hints from the start, more time.',
    },
    normal: {
      temperature: 0.15, candidates: 4, guessSeconds: 35, drawSeconds: 25,
      hintAt: [0.5, 0.75],
      note: 'Clean AI drawings, a letter hint halfway through.',
    },
    hard: {
      temperature: 0.3, candidates: 2, guessSeconds: 25, drawSeconds: 20,
      hintAt: [],
      note: 'Looser AI drawings, no letter hints, less time.',
    },
  };
  // The AI announces a guess once its top word reaches this confidence, and
  // wins the round if that word is right.
  const AI_COMMIT_CONFIDENCE = 0.4;
  const AI_THINK_CONFIDENCE = 0.15; // below this it just says "hmm"
  const CLASSIFY_INTERVAL_MS = 250;
  // Stop sampling more AI drawings once one is this recognizable.
  const CLEAR_ENOUGH_CONFIDENCE = 0.9;

  const $ = (id) => document.getElementById(id);
  const show = (el, visible = true) => el.classList.toggle('hidden', !visible);

  let guesser, artist, board;
  // {settings, roundWords, spareWords, roundIndex, score, results, prepared}
  let game = null;
  let plannedWords = null; // the next game's words, picked early to preload
  let round = null; // state of the round in progress

  // --- Setup -------------------------------------------------------------

  function init() {
    board = new Board($('board'));
    const note = () => {
      const level = document.querySelector('input[name=difficulty]:checked').value;
      $('difficulty-note').textContent = DIFFICULTY[level].note;
    };
    document.querySelectorAll('input[name=difficulty]')
        .forEach((el) => el.addEventListener('change', note));
    note();

    $('start-btn').addEventListener('click', startGame);
    $('again-btn').addEventListener('click', () => showScreen('start'));
    $('guess-form').addEventListener('submit', (e) => {
      e.preventDefault();
      submitGuess();
    });
    $('guess-pass-btn').addEventListener('click', () => endRound('pass'));
    $('draw-pass-btn').addEventListener('click', () => endRound('pass'));
    $('undo-btn').addEventListener('click', () => board.undo());
    $('clear-btn').addEventListener('click', () => {
      board.clear();
      onPlayerDrawingChanged(true);
    });
    board.onChange = (strokeEnded) => onPlayerDrawingChanged(strokeEnded);

    try {
      if (!window.SKETCH_CLASSIFIER_WEIGHTS) {
        throw new Error('js/classifier-weights.js is missing; run training/export.py');
      }
      guesser = new SketchGuesser(window.SKETCH_CLASSIFIER_WEIGHTS);
      artist = new SketchArtist();
      planNextGame();
    } catch (err) {
      $('start-btn').disabled = true;
      $('start-error').textContent = `Can't start: ${err.message}.`;
      show($('start-error'));
    }
  }

  function showScreen(name) {
    for (const s of ['start', 'round', 'results']) show($(`screen-${s}`), s === name);
    show($('round-indicator'), name === 'round');
  }

  function shuffle(list) {
    const a = list.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }

  // Picks the next game's words and starts downloading the first AI round's
  // drawing model while the player is still on the start screen.
  function planNextGame() {
    plannedWords = shuffle(guesser.words);
    artist.preload(plannedWords[0]);
  }

  function startGame() {
    const difficulty = document.querySelector('input[name=difficulty]:checked').value;
    const words = plannedWords;
    game = {
      settings: DIFFICULTY[difficulty],
      roundWords: words.slice(0, ROUNDS),
      spareWords: words.slice(ROUNDS), // replacements if a model won't load
      roundIndex: 0,
      score: 0,
      results: [],
      prepared: new Map(), // word -> Promise of the AI's chosen drawing
    };
    $('score').textContent = '0';
    showScreen('round');
    startRound();
  }

  function roundType(index) {
    return index % 2 === 0 ? 'ai-draws' : 'you-draw';
  }

  // --- Shared round machinery ------------------------------------------

  function startRound() {
    const type = roundType(game.roundIndex);
    round = { type, word: game.roundWords[game.roundIndex], active: false, ended: false };
    $('round-indicator').textContent = `Round ${game.roundIndex + 1} of ${ROUNDS}`;
    show($('guess-panel'), type === 'ai-draws');
    show($('ai-panel'), type === 'you-draw');
    show($('board-tools'), type === 'you-draw');
    board.drawingEnabled = false;
    board.clear();
    const seconds = type === 'ai-draws' ? game.settings.guessSeconds : game.settings.drawSeconds;
    setTimer(seconds, seconds);
    round.nextAiWord =
        game.roundWords[game.roundIndex + (type === 'ai-draws' ? 2 : 1)] || null;
    // The AI sketches its next drawing in the background while you draw.
    if (type === 'you-draw' && round.nextAiWord) prepareAiDrawing(round.nextAiWord);

    if (type === 'ai-draws') startAiDrawsRound();
    else startYouDrawRound();
  }

  function setTimer(remaining, total) {
    $('timer-fill').style.width = `${Math.max(0, remaining / total) * 100}%`;
    $('timer-fill').classList.toggle('low', remaining / total < 0.25);
    $('timer-text').textContent = Math.ceil(Math.max(0, remaining));
  }

  function startTimer(seconds) {
    const timedRound = round;
    round.totalSeconds = seconds;
    round.startedAt = performance.now();
    const tick = () => {
      if (timedRound.ended || timedRound !== round) return;
      const remaining = seconds - (performance.now() - round.startedAt) / 1000;
      setTimer(remaining, seconds);
      if (round.type === 'ai-draws') updateHint(1 - remaining / seconds);
      if (remaining <= 0) {
        endRound('timeout');
        return;
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  function elapsedSeconds() {
    return (performance.now() - round.startedAt) / 1000;
  }

  function pointsFor(elapsed, total) {
    return 50 + Math.round(50 * Math.max(0, 1 - elapsed / total));
  }

  function showOverlay({ icon = '', title = '', text = '', actions = [] }) {
    $('overlay-icon').textContent = icon;
    $('overlay-title').textContent = title;
    $('overlay-text').innerHTML = text;
    const box = $('overlay-actions');
    box.innerHTML = '';
    for (const { label, primary, onClick } of actions) {
      const btn = document.createElement('button');
      btn.className = 'btn' + (primary ? ' primary' : '');
      btn.textContent = label;
      btn.addEventListener('click', onClick);
      box.appendChild(btn);
    }
    show($('overlay'));
    const first = box.querySelector('.primary');
    if (first) first.focus();
  }

  function hideOverlay() {
    show($('overlay'), false);
  }

  function escapeHtml(text) {
    return text.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
  }

  async function endRound(outcome, extra = {}) {
    if (!round || round.ended || !round.active) return;
    const endingRound = round;
    if (round.type === 'you-draw' && outcome !== 'correct') {
      round.active = false; // stops drawing and further guesses
      board.drawingEnabled = false;
      clearTimeout(round.classifyTimer);
      if (outcome === 'timeout' && board.strokes.length) {
        // One last look at the finished drawing.
        const [top] = await guesser.classify(board.strokes, 1);
        if (round !== endingRound || round.ended) return;
        round.best = top;
        if (top.word === round.word && top.confidence >= AI_COMMIT_CONFIDENCE) {
          say(`Just in time! It's ${Words.withArticle(top.word)}! ` +
              `(${Math.round(top.confidence * 100)}% sure)`, 'happy');
          finishRound('correct', { aiGuess: top, aiConfidence: top.confidence });
          return;
        }
      }
      extra = { aiGuess: round.best, aiConfidence: round.best ? round.best.confidence : 0 };
      if (outcome === 'timeout') say("Time's up! I couldn't tell.", 'sad');
    }
    finishRound(outcome, extra);
  }

  function finishRound(outcome, extra) {
    if (round.ended) return;
    round.ended = true;
    round.active = false;
    clearTimeout(round.classifyTimer);
    board.drawingEnabled = false;
    $('guess-input').disabled = true;
    if (round.animation) round.animation.finish();

    const elapsed = elapsedSeconds();
    const success = outcome === 'correct';
    const points = success ? pointsFor(elapsed, round.totalSeconds) : 0;
    game.score += points;
    $('score').textContent = game.score;

    const word = Words.display(round.word);
    const result = {
      type: round.type, word: round.word, success, points,
      seconds: elapsed, thumb: board.snapshot(), ...extra,
    };
    game.results.push(result);

    let icon, title, text;
    if (round.type === 'ai-draws') {
      revealHint();
      $('guess-feedback').textContent = '';
      icon = success ? '🎉' : '⏱️';
      title = success ? `Yes! It's ${Words.withArticle(round.word)}.` :
          outcome === 'pass' ? 'Passed.' : "Time's up!";
      text = success ?
          `You got it in ${elapsed.toFixed(1)}s. <b>+${points}</b>` :
          `The AI was drawing <b>${escapeHtml(Words.withArticle(round.word))}</b>.`;
    } else {
      const best = extra.aiGuess;
      icon = success ? '🤖' : '🤷';
      title = success ? `The AI got it: ${word}!` :
          outcome === 'pass' ? 'Passed.' : "Time's up!";
      text = success ?
          `It was ${Math.round(extra.aiConfidence * 100)}% sure after ${elapsed.toFixed(1)}s. <b>+${points}</b>` :
          best ? `The AI's best guess was <b>${escapeHtml(Words.display(best.word))}</b> ` +
                 `(${Math.round(best.confidence * 100)}%).` :
                 "The AI didn't see enough to guess.";
    }
    const last = game.roundIndex === ROUNDS - 1;
    showOverlay({
      icon, title, text,
      actions: [{
        label: last ? 'See results' : 'Next round →',
        primary: true,
        onClick: nextRound,
      }],
    });
  }

  function nextRound() {
    hideOverlay();
    game.roundIndex++;
    if (game.roundIndex >= ROUNDS) showResults();
    else startRound();
  }

  // --- AI draws, you guess ---------------------------------------------

  async function startAiDrawsRound() {
    const settings = game.settings;
    $('round-type').textContent = 'AI draws · you guess';
    $('prompt').textContent = 'Name the AI\'s drawing before time runs out';
    $('guess-input').value = '';
    $('guess-input').disabled = true;
    $('guess-feedback').textContent = '';
    $('guess-list').innerHTML = '';
    round.revealed = new Set();
    updateHint(0);
    showOverlay({
      icon: '✏️', title: 'The AI is picking up its pen…',
      text: game.prepared.has(round.word) ? 'Finishing its sketch.' :
          'Downloading its drawing model and trying a few sketches.',
    });

    let drawing;
    const myRound = round;
    try {
      drawing = await prepareAiDrawing(round.word);
    } catch (err) {
      game.prepared.delete(myRound.word);
      if (round !== myRound) return;
      console.error(err);
      showOverlay({
        icon: '📡', title: "Couldn't load the AI's drawing model",
        text: 'The drawing models are downloaded from Google Cloud Storage. ' +
              'Check your internet connection.',
        actions: [
          { label: 'Try again', primary: true, onClick: () => startAiDrawsRound() },
          { label: 'Different word', onClick: () => {
            round.word = game.roundWords[game.roundIndex] = game.spareWords.pop();
            startAiDrawsRound();
          } },
        ],
      });
      return;
    }
    if (round !== myRound) return;
    artist.keepOnly([round.nextAiWord]);
    if (round.nextAiWord) prepareAiDrawing(round.nextAiWord);
    hideOverlay();

    round.active = true;
    const duration = Math.min(10000, Math.max(3500, drawing.segments.length * 45));
    round.animation = board.animate(drawing.segments, duration);
    $('guess-input').disabled = false;
    $('guess-input').focus();
    startTimer(settings.guessSeconds);
  }

  // Starts (once per word) sampling drawings in the background and resolves
  // to the one the classifier recognizes best, so the player gets the
  // clearest sketch rather than whichever came out first.
  function prepareAiDrawing(word) {
    if (!game.prepared.has(word)) {
      const { temperature, candidates } = game.settings;
      const best = (async () => {
        let chosen = null;
        for (let i = 0; i < candidates; i++) {
          const [drawing] = await artist.drawMany(word, temperature, 1);
          if (drawing.segments.length < 4) continue;
          const probs = await guesser.probabilities(drawing.strokes);
          drawing.targetConfidence = probs[guesser.words.indexOf(word)];
          if (!chosen || drawing.targetConfidence > chosen.targetConfidence) chosen = drawing;
          if (chosen.targetConfidence >= CLEAR_ENOUGH_CONFIDENCE) break;
        }
        if (!chosen) throw new Error(`no usable drawing for ${word}`);
        return chosen;
      })();
      best.catch(() => {}); // reported when the round awaits it
      game.prepared.set(word, best);
    }
    return game.prepared.get(word);
  }

  function updateHint(progress) {
    const word = Words.display(round.word);
    const letters = [...word].map((ch, i) => (ch === ' ' ? -1 : i)).filter((i) => i >= 0);
    const due = game.settings.hintAt.filter((t) => progress >= t).length;
    while (round.revealed.size < Math.min(due, letters.length - 1)) {
      const hidden = letters.filter((i) => !round.revealed.has(i));
      round.revealed.add(round.revealed.size === 0 ? letters[0] :
                         hidden[Math.floor(Math.random() * hidden.length)]);
    }
    const key = [...round.revealed].join(',');
    if (key === round.hintKey) return;
    round.hintKey = key;
    const hint = $('hint');
    hint.innerHTML = '';
    let group = null; // one group per word, so lines only break between words
    [...word].forEach((ch, i) => {
      if (ch === ' ') {
        group = null;
        return;
      }
      if (!group) {
        group = document.createElement('span');
        group.className = 'hint-word';
        hint.appendChild(group);
      }
      const span = document.createElement('span');
      span.className = 'letter';
      span.textContent = round.revealed.has(i) ? ch : '';
      group.appendChild(span);
    });
    const count = document.createElement('div');
    count.className = 'hint-count';
    const words = word.split(' ').length;
    count.textContent = words > 1 ? `${words} words` : `${letters.length} letters`;
    hint.appendChild(count);
  }

  function revealHint() {
    const letters = Words.display(round.word).replace(/ /g, '');
    document.querySelectorAll('#hint .letter').forEach((span, i) => {
      span.textContent = letters[i];
    });
  }

  function submitGuess() {
    if (!round || !round.active || round.type !== 'ai-draws') return;
    const input = $('guess-input');
    const guess = input.value.trim();
    if (!guess) return;
    input.value = '';
    const verdict = Words.checkGuess(guess, round.word);
    if (verdict === 'correct') {
      endRound('correct');
      return;
    }
    const feedback = $('guess-feedback');
    feedback.textContent = verdict === 'close' ? 'So close! Check the spelling.' : 'Nope, keep guessing.';
    feedback.className = 'feedback ' + verdict;
    const li = document.createElement('li');
    li.textContent = guess;
    $('guess-list').prepend(li);
  }

  // --- You draw, AI guesses --------------------------------------------

  function startYouDrawRound() {
    const settings = game.settings;
    const word = Words.display(round.word);
    $('round-type').textContent = 'You draw · AI guesses';
    $('prompt').innerHTML = `Draw: <b>${escapeHtml(word)}</b>`;
    $('ai-guesses').innerHTML = '';
    say("Ready when you are.", 'idle');
    round.best = null;
    showOverlay({
      icon: '✏️',
      title: `Your turn: draw ${Words.withArticle(round.word)}`,
      text: `The AI has <b>${settings.drawSeconds} seconds</b> to recognize it. ` +
            'No letters or numbers!',
      actions: [{
        label: 'Start drawing', primary: true, onClick: () => {
          hideOverlay();
          round.active = true;
          board.drawingEnabled = true;
          startTimer(settings.drawSeconds);
        },
      }],
    });
  }

  function say(text, mood) {
    const bubble = $('ai-bubble');
    bubble.textContent = text;
    bubble.className = 'bubble ' + (mood || '');
    $('ai-avatar').className = 'ai-avatar ' + (mood || '');
  }

  function onPlayerDrawingChanged(strokeEnded) {
    if (!round || !round.active || round.type !== 'you-draw') return;
    if (strokeEnded) {
      clearTimeout(round.classifyTimer);
      round.classifyTimer = null;
      classifyPlayerDrawing();
    } else if (!round.classifyTimer) {
      round.classifyTimer = setTimeout(() => {
        round.classifyTimer = null;
        classifyPlayerDrawing();
      }, CLASSIFY_INTERVAL_MS);
    }
  }

  // Asks the guesser thread about the current drawing. If a guess is
  // already being computed, only the newest drawing is looked at next.
  async function classifyPlayerDrawing() {
    const myRound = round;
    if (!myRound.active) return;
    if (myRound.guessing) {
      myRound.guessAgain = true;
      return;
    }
    const strokes = board.strokes.map(([xs, ys]) => [xs.slice(), ys.slice()]);
    const points = strokes.reduce((n, [xs]) => n + xs.length, 0);
    if (points < 3) {
      renderGuessBars([]);
      say('Ready when you are.', 'idle');
      return;
    }
    myRound.guessing = true;
    let guesses;
    try {
      guesses = await guesser.classify(strokes, 5);
    } finally {
      myRound.guessing = false;
    }
    if (round !== myRound || !myRound.active) return;
    if (myRound.guessAgain) {
      myRound.guessAgain = false;
      classifyPlayerDrawing();
    }
    renderGuessBars(guesses);
    const top = guesses[0];
    myRound.best = top;
    const pct = Math.round(top.confidence * 100);
    if (top.confidence >= AI_COMMIT_CONFIDENCE) {
      if (top.word === myRound.word) {
        say(`I know! It's ${Words.withArticle(top.word)}! (${pct}% sure)`, 'happy');
        finishRound('correct', { aiGuess: top, aiConfidence: top.confidence });
        return;
      }
      say(`Is it ${Words.withArticle(top.word)}? (${pct}%)`, 'asking');
    } else if (top.confidence >= AI_THINK_CONFIDENCE) {
      say(`Hmm… maybe ${Words.display(top.word)}? (${pct}%)`, 'thinking');
    } else {
      say('Hmm… no idea yet.', 'thinking');
    }
  }

  function renderGuessBars(guesses) {
    const list = $('ai-guesses');
    list.innerHTML = '';
    for (const g of guesses) {
      const li = document.createElement('li');
      if (g.word === round.word) li.classList.add('target');
      const pct = Math.round(g.confidence * 100);
      li.innerHTML = `<span class="bar-label"></span><span class="bar-pct">${pct}%</span>` +
                     `<span class="bar"><span style="width:${Math.max(pct, 1)}%"></span></span>`;
      li.querySelector('.bar-label').textContent = Words.display(g.word);
      list.appendChild(li);
    }
  }

  // --- Results ---------------------------------------------------------

  function showResults() {
    showScreen('results');
    planNextGame();
    $('final-score').textContent = game.score;
    const ai = game.results.filter((r) => r.type === 'ai-draws');
    const you = game.results.filter((r) => r.type === 'you-draw');
    $('summary-you').textContent = `${ai.filter((r) => r.success).length} / ${ai.length}`;
    $('summary-ai').textContent = `${you.filter((r) => r.success).length} / ${you.length}`;
    const gallery = $('gallery');
    gallery.innerHTML = '';
    for (const r of game.results) {
      const fig = document.createElement('figure');
      fig.className = r.success ? 'win' : 'miss';
      const img = document.createElement('img');
      img.src = r.thumb;
      img.alt = `${r.type === 'ai-draws' ? "AI's" : 'Your'} drawing of ${Words.withArticle(r.word)}`;
      const cap = document.createElement('figcaption');
      const who = r.type === 'ai-draws' ? 'AI drew' : 'You drew';
      let detail;
      if (r.type === 'ai-draws') {
        detail = r.success ? `you guessed it in ${r.seconds.toFixed(1)}s` : 'you missed it';
      } else if (r.success) {
        detail = `AI got it, ${Math.round(r.aiConfidence * 100)}% sure`;
      } else {
        detail = r.aiGuess ? `AI guessed ${Words.display(r.aiGuess.word)}` : 'AI had no guess';
      }
      cap.innerHTML = `<b></b><span></span>`;
      cap.querySelector('b').textContent = `${who}: ${Words.display(r.word)}`;
      cap.querySelector('span').textContent = `${detail}${r.points ? ` · +${r.points}` : ''}`;
      fig.append(img, cap);
      gallery.appendChild(fig);
    }
  }

  init();
})();
