"""Assign lyric units (words or syllables) to detected notes.

Segment-level dynamic programming: each unit takes a run of consecutive notes. The cost of
giving unit j the notes k+1..i combines
  * timing    distance between the first note and the unit's rough time (Whisper), capped,
              so drifting timestamps can't drag everything off;
  * syllables a k-syllable word wants k notes (extra notes = melisma, cheap; too few = costly);
  * phrasing  lyric lines want to start after a rest, and words rarely continue across one.
Notes no unit claims become vocalise notes (placeholder syllable). Units with no note are
repaired afterwards by splitting the note sounding at their time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lyrics import Unit
from .notes import Note


@dataclass
class AssignParams:
    sigma: float = 0.25        # seconds of timing error that cost 1 (looser when timings are rough)
    timing_cap: float = 4.0
    timing_weight: float = 1.0
    too_few: float = 1.5       # per missing note for a multi-syllable word
    melisma: float = 0.35      # per extra note
    inner_rest: float = 2.5    # per short rest (> rest_s) inside one unit
    max_span_rest_s: float = 0.45  # a unit never continues across a rest this long
    line_no_rest: float = 1.2  # a lyric line starting mid-phrase
    word_after_long_rest: float = 0.8
    rest_s: float = 0.3
    long_rest_s: float = 0.8
    orphan: float = 2.6        # per note left without a unit
    far_s: float = 2.5         # notes this far from every trusted word time are cheap to leave wordless
    far_orphan: float = 0.8
    edge_orphan: float = 0.15  # before the first / after the last trusted word: lines the sheet doesn't have
    skip: float = 6.0          # per unit left without a note
    max_notes: int = 10        # longest run of notes one unit may take
    band_s: float = 20.0       # ignore pairings whose times are further apart than this


def assign_lyrics(notes: list[Note], units: list[Unit], fill: str, ext: str = "+",
                  params: AssignParams | None = None, repair: bool = True) -> tuple[list[Note], dict]:
    """repair=False for harmony lines: they sing only some of the words, so unplaced words are fine."""
    p = params or AssignParams()
    notes = sorted(notes, key=lambda n: n.start)
    N, M = len(notes), len(units)
    stats = {"notes": N, "units": M, "orphans": 0, "skipped": 0, "repaired": 0}
    if N == 0:
        return notes, stats
    if M == 0:
        _label_orphans(notes, list(range(N)), fill, ext)
        stats["orphans"] = N
        return notes, stats

    t = np.array([n.start for n in notes])
    e = np.array([n.end for n in notes])
    gap = np.concatenate([[np.inf], t[1:] - e[:-1]])           # rest before note i
    rest_cum = np.concatenate([[0], np.cumsum(gap[1:] > p.rest_s)])  # rests before notes 1..i
    long_cum = np.concatenate([[0], np.cumsum(gap[1:] > p.max_span_rest_s)])
    anchor = np.array([u.start for u in units])
    wts = np.array([getattr(u, "weight", 1.0) for u in units])
    syl = np.array([max(1, getattr(u, "syllables", 1)) for u in units])
    line0 = np.array([bool(getattr(u, "line_start", False)) for u in units])
    first = np.array([u.first_in_word for u in units])

    trusted = anchor[wts >= 0.3] if np.any(wts >= 0.3) else anchor
    srt = np.sort(trusted)
    pos = np.clip(np.searchsorted(srt, t), 1, len(srt) - 1) if len(srt) > 1 else np.zeros(N, int)
    dist = np.minimum(np.abs(t - srt[pos - 1]), np.abs(t - srt[pos])) if len(srt) > 1 else np.abs(t - srt[0])
    orphan_cost = np.where(dist > p.far_s, p.far_orphan, p.orphan)
    outside = (t > srt[-1] + p.far_s) | (t < srt[0] - p.far_s)
    orphan_cost = np.where(outside, p.edge_orphan, orphan_cost)

    INF = 1e18
    dp = np.full((M + 1, N + 1), INF)
    # back[j, i] = (kind, k): kind 0 = orphan note i, 1 = skip unit j, 2 = unit j covers notes k+1..i
    bk_kind = np.zeros((M + 1, N + 1), dtype=np.int8)
    bk_k = np.zeros((M + 1, N + 1), dtype=np.int32)
    dp[0, 0] = 0.0
    for i in range(1, N + 1):
        dp[0, i] = dp[0, i - 1] + orphan_cost[i - 1]
        bk_kind[0, i] = 0

    for j in range(1, M + 1):
        u = j - 1
        lo_t, hi_t = anchor[u] - p.band_s, anchor[u] + p.band_s
        for i in range(0, N + 1):
            best, kind, kk = dp[j - 1, i] + p.skip, 1, 0
            if i > 0:
                c = dp[j, i - 1] + orphan_cost[i - 1]
                if c < best:
                    best, kind, kk = c, 0, 0
                for k in range(max(0, i - p.max_notes), i):
                    prev = dp[j - 1, k]
                    if prev >= INF:
                        continue
                    s = k  # first note index of the segment (0-based) is k
                    if wts[u] >= 0.5 and not (lo_t <= t[s] <= hi_t):
                        continue  # trusted timing: don't search absurdly far away
                    if long_cum[i - 1] - long_cum[s] > 0:
                        continue  # never smear one word across a real rest
                    n = i - k
                    cost = p.timing_weight * wts[u] * min(p.timing_cap, abs(t[s] - anchor[u]) / p.sigma)
                    if first[u]:
                        if n < syl[u]:
                            cost += (syl[u] - n) * p.too_few
                        elif n > syl[u]:
                            cost += (n - syl[u]) * p.melisma
                    else:
                        cost += (n - 1) * p.melisma
                    cost += (rest_cum[i - 1] - rest_cum[s]) * p.inner_rest
                    if line0[u] and gap[s] < 0.2 and s > 0:
                        cost += p.line_no_rest
                    elif not line0[u] and first[u] and gap[s] > p.long_rest_s and s > 0:
                        cost += p.word_after_long_rest
                    c = prev + cost
                    if c < best:
                        best, kind, kk = c, 2, k
            dp[j, i], bk_kind[j, i], bk_k[j, i] = best, kind, kk

    # trace back from (M, N)
    owner = [-1] * N
    used = set()
    j, i = M, N
    while j > 0 or i > 0:
        kind = bk_kind[j, i] if j > 0 else 0
        if j == 0 or kind == 0:
            i -= 1
        elif kind == 1:
            j -= 1
        else:
            k = int(bk_k[j, i])
            for q in range(k, i):
                owner[q] = j - 1
            used.add(j - 1)
            j, i = j - 1, k

    orphans = []
    prev_owner = None
    for idx, n in enumerate(notes):
        o = owner[idx]
        if o < 0:
            orphans.append(idx)
            prev_owner = None
            continue
        if o != prev_owner:
            n.lyric, n.unit, n.is_unit_start = units[o].text, o, True
        else:
            n.lyric, n.unit = ext, o
        prev_owner = o
    _label_orphans(notes, orphans, fill, ext)
    stats["orphans"] = len(orphans)
    skipped = [k for k in range(M) if k not in used]
    stats["skipped"] = len(skipped)
    notes, repaired = _repair_skipped(notes, units, skipped) if repair else (notes, 0)
    stats["repaired"] = repaired
    notes = merge_continuations(notes, ext)
    notes = tidy_ornaments(notes, ext)
    return notes, stats


def tidy_ornaments(notes: list[Note], ext: str = "+", scoop_s: float = 0.15, blip_s: float = 0.10,
                   max_jump: float = 4.5) -> list[Note]:
    """Fold sung ornaments back into the notes they decorate, the way a person would chart them:
      * scoops: a short first note of a word sliding into a longer note -> one note at the target pitch
      * blips:  a very short '+' passing note inside a slide -> absorbed by its neighbour
    Fast runs where every note is short are left alone (the 'longer neighbour' rule)."""
    out = list(notes)
    changed = True
    while changed:
        changed = False
        for i, n in enumerate(out):
            nxt = out[i + 1] if i + 1 < len(out) else None
            prv = out[i - 1] if i > 0 else None
            glued_next = nxt is not None and nxt.start - n.end < 0.05
            glued_prev = prv is not None and n.start - prv.end < 0.05
            if (n.lyric != ext and glued_next and nxt.lyric == ext and n.dur < scoop_s
                    and nxt.dur >= 1.8 * n.dur and abs(nxt.pitch - n.pitch) <= max_jump):
                merged = Note(n.start, nxt.end, nxt.pitch, n.lyric, n.unit, n.is_unit_start, n.forced)
                out[i:i + 2] = [merged]
                changed = True
                break
            if n.lyric == ext and n.dur < blip_s and (glued_prev or glued_next):
                cands = []
                if glued_prev and abs(prv.pitch - n.pitch) <= max_jump:
                    cands.append((abs(prv.pitch - n.pitch), "p"))
                if glued_next and nxt.lyric == ext and abs(nxt.pitch - n.pitch) <= max_jump:
                    cands.append((abs(nxt.pitch - n.pitch), "n"))
                if not cands:
                    continue
                if min(cands)[1] == "p":
                    prv.end = n.end
                    out.pop(i)
                else:
                    nxt.start = n.start
                    out.pop(i)
                changed = True
                break
    return out


def ensure_syllable_notes(notes: list[Note], units: list[Unit], ext: str = "+", rms_db=None, hop: float = 0.01,
                          min_piece: float = 0.07, borrow_s: float = 0.12) -> tuple[list[Note], int]:
    """Give every syllable of a word its own note.

    When a word's syllables are sung on one steady pitch, segmentation finds one note and the
    phonemizer has to cram 'mel-an-cho-ly' into it. Split the word's longest notes (at loudness dips
    between syllables when there are any, otherwise evenly) until notes >= syllables. A very short
    word may borrow a little time from a following gap first. Returns (notes, words fixed)."""
    notes = sorted(notes, key=lambda n: n.start)
    fixed = 0
    i = 0
    while i < len(notes):
        n = notes[i]
        if not n.is_unit_start or n.unit < 0 or n.unit >= len(units):
            i += 1
            continue
        syl = max(1, getattr(units[n.unit], "syllables", 1))
        j = i + 1
        while j < len(notes) and notes[j].lyric == ext and notes[j].unit == n.unit:
            j += 1
        group = notes[i:j]
        if len(group) >= syl:
            i = j
            continue
        # borrow time from a gap after the word, if the word is too short for its syllables
        nxt_start = notes[j].start if j < len(notes) else group[-1].end + borrow_s * syl
        need = syl * min_piece - sum(g.dur for g in group)
        if need > 0:
            group[-1].end = min(nxt_start, group[-1].end + min(need, borrow_s * syl))
        changed = True
        while len(group) < syl and changed:
            changed = False
            host = max(range(len(group)), key=lambda k: group[k].dur)
            h = group[host]
            if h.dur < 2 * min_piece:
                break
            cut = h.start + h.dur / 2
            if rms_db is not None:
                edge = max(min_piece, 0.25 * h.dur)  # a dip right after the attack is the consonant, not a syllable
                a = int((h.start + edge) / hop)
                b = int((h.end - edge) / hop)
                if b > a + 1:
                    seg = rms_db[a:b]
                    k = int(np.argmin(seg))
                    if seg.max() - seg[k] >= 3.0:  # a real dip between syllables
                        cut = (a + k) * hop
            left = Note(h.start, cut, h.pitch, h.lyric, h.unit, h.is_unit_start, h.forced)
            right = Note(cut, h.end, h.pitch, ext, h.unit, False, True)
            group[host:host + 1] = [left, right]
            changed = True
        if len(group) > j - i:
            fixed += 1
        notes[i:j] = group
        i += len(group)
    return _rebalance_syllables(notes, ext, rms_db, hop), fixed


def _rebalance_syllables(notes: list[Note], ext: str, rms_db, hop: float, short: float = 0.16,
                         ratio: float = 2.0) -> list[Note]:
    """Two syllables of one word on the same pitch where one got a sliver (82 ms 'grow' + 326 ms 'ing'):
    move the boundary to the loudness dip in the middle of the pair, or halfway."""
    for k in range(len(notes) - 1):
        a, b = notes[k], notes[k + 1]
        if b.lyric != ext or b.unit != a.unit or a.unit < 0 or a.tone != b.tone or b.start - a.end > 0.02:
            continue
        lo, hi = min(a.dur, b.dur), max(a.dur, b.dur)
        if lo >= short or hi < ratio * lo:
            continue
        s, e = a.start, b.end
        cut = 0.5 * (s + e)
        if rms_db is not None:
            i, j = int((s + 0.3 * (e - s)) / hop), int((s + 0.7 * (e - s)) / hop)
            if j > i + 1:
                seg = rms_db[i:j]
                if seg.max() - seg.min() >= 3.0:
                    cut = (i + int(np.argmin(seg))) * hop
        a.end = b.start = cut
    return notes


def vowel_onsets(notes: list[Note], lang: str, ext: str = "+") -> list[Note]:
    """Move word-start notes later by their voiced initial consonant, so the vowel (not the 'm' of
    'my') sits on the note and UTAU's own consonant placement lands where the singer's did."""
    from .syllables import onset_consonant_s
    for n in notes:
        if not n.is_unit_start or not n.lyric or n.lyric.startswith(ext):
            continue
        d = min(onset_consonant_s(n.lyric, lang), 0.4 * n.dur)
        if d > 0.005:
            n.start += d
    return notes


def min_syllable_length(notes: list[Note], ext: str = "+", min_s: float = 0.14) -> list[Note]:
    """A syllable squeezed onto a sliver can't be sung by the voicebank: lengthen it into the next syllable
    of the same word when that one is on the same pitch and can spare the time."""
    notes = sorted(notes, key=lambda n: n.start)
    for a, b in zip(notes, notes[1:]):
        if a.dur >= min_s or b.lyric != ext or b.unit != a.unit or a.unit < 0 or b.start - a.end > 0.02:
            continue
        if a.tone != b.tone:  # a pitch change stays where it was sung
            continue
        need = min_s - a.dur
        if b.dur - need >= min_s:
            a.end += need
            b.start = a.end
    return notes


def legato(notes: list[Note], max_gap: float = 0.3) -> list[Note]:
    """Close the small gaps unvoiced consonants leave between notes. In UTAU the phonemizer puts
    each consonant before its note, so touching notes sound connected; gaps sound chopped."""
    notes = sorted(notes, key=lambda n: n.start)
    for a, b in zip(notes, notes[1:]):
        g = b.start - a.end
        if 0 < g < max_gap:
            a.end = b.start
    return notes


def _label_orphans(notes: list[Note], idx: list[int], fill: str, ext: str):
    s = set(idx)
    for i in idx:
        n = notes[i]
        glued = i > 0 and (i - 1) in s and n.start - notes[i - 1].end < 0.08
        n.lyric = ext if glued else fill
        n.unit = -1


def _repair_skipped(notes: list[Note], units: list[Unit], skipped: list[int]) -> tuple[list[Note], int]:
    """Give each skipped unit a note: split the note its predecessor ended on (or the one sounding
    at its time) so the unit gets the second half."""
    repaired = 0
    for k in skipped:
        u = units[k]
        # prefer the last note owned by the previous unit, so word order is preserved
        host = None
        for idx in range(len(notes) - 1, -1, -1):
            if notes[idx].unit == k - 1 and k > 0:
                host = idx
                break
        if host is None:
            for idx, n in enumerate(notes):
                if n.start <= u.start <= n.end:
                    host = idx
                    break
        if host is None:
            continue
        n = notes[host]
        if n.dur < 0.12:
            continue
        cut = n.start + n.dur / 2
        a = Note(n.start, cut, n.pitch, n.lyric, n.unit, n.is_unit_start, n.forced)
        b = Note(cut, n.end, n.pitch, u.text, k, True, True)
        notes[host:host + 1] = [a, b]
        repaired += 1
    return notes, repaired


def merge_continuations(notes: list[Note], ext: str = "+") -> list[Note]:
    """A '+' note on the same pitch right after its parent adds nothing; fold it in."""
    out: list[Note] = []
    for n in notes:
        if out and n.lyric == ext and n.tone == out[-1].tone and n.start - out[-1].end < 0.03:
            out[-1].end = n.end
            if n.contour is not None and out[-1].contour is not None:
                out[-1].contour = np.concatenate([out[-1].contour, n.contour])
        else:
            out.append(n)
    return out
