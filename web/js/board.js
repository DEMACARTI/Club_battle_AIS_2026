// The drawing board: a canvas with a fixed 640x480 coordinate system that
// scales to its on-screen size. Used both for the AI's animated drawings and
// for the player's freehand strokes.
(function (root) {
  'use strict';

  const W = 640;
  const H = 480;
  const INK = '#1d2433';
  const LINE = 5;
  const MIN_POINT_GAP = 2; // ignore pointer jitter smaller than this

  class Board {
    constructor(canvas) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.strokes = []; // player strokes: [[xs], [ys]]
      this.onChange = null; // called while the player draws
      this.drawingEnabled = false;
      this.activeStroke = null;
      this.resize();
      root.addEventListener('resize', () => this.resize());
      this.bindPointer();
    }

    resize() {
      const dpr = root.devicePixelRatio || 1;
      this.canvas.width = W * dpr;
      this.canvas.height = H * dpr;
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.redraw();
    }

    clear() {
      this.strokes = [];
      this.segments = null;
      this.segmentCount = 0;
      this.redraw();
    }

    redraw() {
      const ctx = this.ctx;
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, W, H);
      ctx.strokeStyle = INK;
      ctx.fillStyle = INK;
      ctx.lineWidth = LINE;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      for (const [xs, ys] of this.strokes) this.drawPolyline(xs, ys);
      if (this.segments) {
        for (let i = 0; i < this.segmentCount; i++) this.drawSegment(this.segments[i]);
      }
    }

    drawPolyline(xs, ys) {
      const ctx = this.ctx;
      ctx.beginPath();
      if (xs.length === 1) {
        ctx.arc(xs[0], ys[0], LINE / 2, 0, 2 * Math.PI);
        ctx.fill();
        return;
      }
      ctx.moveTo(xs[0], ys[0]);
      for (let i = 1; i < xs.length; i++) ctx.lineTo(xs[i], ys[i]);
      ctx.stroke();
    }

    drawSegment([x0, y0, x1, y1]) {
      const ctx = this.ctx;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
    }

    // --- AI drawing -------------------------------------------------------

    // Scales and centers segments ([x0, y0, x1, y1], any units) to fill the board.
    fitSegments(segments) {
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      for (const [x0, y0, x1, y1] of segments) {
        minX = Math.min(minX, x0, x1); maxX = Math.max(maxX, x0, x1);
        minY = Math.min(minY, y0, y1); maxY = Math.max(maxY, y0, y1);
      }
      const pad = 48;
      const scale = Math.min((W - 2 * pad) / Math.max(maxX - minX, 1e-6),
                             (H - 2 * pad) / Math.max(maxY - minY, 1e-6));
      const offX = (W - (maxX - minX) * scale) / 2 - minX * scale;
      const offY = (H - (maxY - minY) * scale) / 2 - minY * scale;
      return segments.map(([x0, y0, x1, y1]) => [
        x0 * scale + offX, y0 * scale + offY, x1 * scale + offX, y1 * scale + offY]);
    }

    // Animates segments onto the board. Returns {done: Promise, finish()}.
    animate(segments, durationMs) {
      this.clear();
      this.segments = this.fitSegments(segments);
      const total = this.segments.length;
      const start = performance.now();
      let finished = false;
      let resolveDone;
      const done = new Promise((resolve) => { resolveDone = resolve; });
      const step = (now) => {
        if (finished) return;
        const target = Math.min(total,
                                Math.ceil(total * (now - start) / durationMs));
        while (this.segmentCount < target) {
          this.drawSegment(this.segments[this.segmentCount++]);
        }
        if (this.segmentCount >= total) {
          finished = true;
          resolveDone();
          return;
        }
        root.requestAnimationFrame(step);
      };
      root.requestAnimationFrame(step);
      return {
        done,
        finish: () => {
          if (finished) return;
          finished = true;
          this.segmentCount = total;
          this.redraw();
          resolveDone();
        },
      };
    }

    // --- Player drawing ---------------------------------------------------

    toBoardPoint(event) {
      // Cached briefly: layout reads are not free and pointermove is hot.
      const now = performance.now();
      if (!this.rect || now - this.rectTime > 100) {
        this.rect = this.canvas.getBoundingClientRect();
        this.rectTime = now;
      }
      const rect = this.rect;
      return [(event.clientX - rect.left) * W / rect.width,
              (event.clientY - rect.top) * H / rect.height];
    }

    bindPointer() {
      const c = this.canvas;
      c.addEventListener('pointerdown', (e) => {
        if (!this.drawingEnabled || e.button > 0) return;
        e.preventDefault();
        c.setPointerCapture(e.pointerId);
        const [x, y] = this.toBoardPoint(e);
        this.activeStroke = [[x], [y]];
        this.strokes.push(this.activeStroke);
        this.drawPolyline([x], [y]);
        if (this.onChange) this.onChange(false);
      });
      c.addEventListener('pointermove', (e) => {
        if (!this.activeStroke) return;
        // Browsers deliver at most one pointermove per frame; the positions
        // in between come as coalesced events. Using them keeps fast strokes
        // smooth instead of turning curves into straight segments.
        const events = e.getCoalescedEvents ? e.getCoalescedEvents() : [];
        const [xs, ys] = this.activeStroke;
        let added = false;
        for (const ev of events.length ? events : [e]) {
          const [x, y] = this.toBoardPoint(ev);
          const lx = xs[xs.length - 1], ly = ys[ys.length - 1];
          if (Math.hypot(x - lx, y - ly) < MIN_POINT_GAP) continue;
          xs.push(x);
          ys.push(y);
          this.drawSegment([lx, ly, x, y]);
          added = true;
        }
        if (added && this.onChange) this.onChange(false);
      });
      const end = () => {
        if (!this.activeStroke) return;
        this.activeStroke = null;
        if (this.onChange) this.onChange(true);
      };
      c.addEventListener('pointerup', end);
      c.addEventListener('pointercancel', end);
    }

    undo() {
      this.strokes.pop();
      this.redraw();
      if (this.onChange) this.onChange(true);
    }

    // Small PNG of the board for the results screen. Encoding the full-size
    // (devicePixelRatio-scaled) canvas would stall the page for a moment.
    snapshot() {
      if (!this.thumb) {
        this.thumb = document.createElement('canvas');
        this.thumb.width = 320;
        this.thumb.height = 240;
      }
      const ctx = this.thumb.getContext('2d');
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(this.canvas, 0, 0, 320, 240);
      return this.thumb.toDataURL('image/png');
    }
  }

  root.Board = Board;
})(window);
