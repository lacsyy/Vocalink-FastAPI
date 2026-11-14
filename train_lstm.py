"""
Train the AAC next-icon LSTM and write lstm_weights.npz.

Training data:
  1) Real sentences students sent from the AAC board (aac_logs.icon_sequence),
     read from DATABASE_URL (or ./vocalink.db by default).
  2) A hand-written seed set of common classroom AAC sentences, so the model
     is usable before enough real data exists. Real sentences are repeated
     REAL_WEIGHT times so they dominate once collected.

Usage:
  python train_lstm.py                 # seed + database
  python train_lstm.py --no-db         # seed only
  python train_lstm.py --epochs 60 --out lstm_weights.npz

Besides the weights, it writes results/training_results.md and .json: held-out
top-1/top-3 accuracy (LSTM vs. bigram and unigram baselines, mean ± std over
several random splits), perplexity, the loss curve and sample predictions.

Restart the API afterwards (or redeploy) so it loads the new weights.
"""

import argparse
import datetime as dt
import json
import math
import os
import random
import time
from collections import Counter, defaultdict

from lstm_model import NextIconLSTM

REAL_WEIGHT = 3

# Icons that can appear inside a built sentence. Request icons (medicine,
# toilet, emergency, hurt, ...) are excluded: the board sends them as
# requests immediately instead of adding them to the sentence.
ICON_IDS = [
    "water", "food", "rest", "bag", "cold", "hot", "too_loud", "pencil", "tissue",
    "need_break", "go_outside", "fix_chair",
    "happy", "sad", "sick", "scared", "angry", "confused", "excited", "tired", "bored",
    "worried", "frustrated", "calm", "proud", "embarrassed",
    "help", "question", "done", "repeat", "teacher", "understand", "not_understand",
    "more_time", "write", "read", "group_work", "worksheet", "computer", "show_me",
    "notebook", "sit_down", "line_up", "activity",
    "yes", "no", "please", "thankyou", "wait", "stop", "more_a", "give_me", "come_here",
    "open_it", "finished", "go_a", "try_it", "again",
    "math_plus", "math_minus", "math_times", "math_divide", "math_equals", "math_numbers",
    "math_count", "math_shapes", "math_measure", "math_more", "math_less", "math_answer",
    "math_time", "math_money", "math_pattern", "math_bigger", "math_smaller", "math_hundred",
    "num_0", "num_1", "num_2", "num_3", "num_4", "num_5", "num_6", "num_7", "num_8",
    "num_9", "num_10",
    "comm_hello", "comm_sorry", "comm_excuse", "comm_goodjob", "comm_myturn",
    "comm_yourturn", "comm_talk", "comm_listen", "comm_louder", "comm_slower",
    "comm_goodbye", "comm_good_morning", "comm_i_want", "comm_i_like", "comm_i_dont_like",
    "comm_friend", "comm_agree", "comm_disagree",
    "people_me", "people_teacher", "people_friend", "people_classmate", "people_mom",
    "people_dad", "people_nurse", "people_principal", "people_aide",
    "places_classroom", "places_bathroom", "places_cafeteria", "places_library", "places_gym",
    "places_playground", "places_office", "places_hallway", "places_home",
    "flag_au", "flag_br", "flag_ca", "flag_cn", "flag_de", "flag_eg", "flag_es", "flag_fr",
    "flag_gb", "flag_id", "flag_in", "flag_it", "flag_jp", "flag_kr", "flag_mx", "flag_my",
    "flag_ng", "flag_ph", "flag_ru", "flag_sg", "flag_th", "flag_us", "flag_vn", "flag_za",
]
FLAG_IDS = [i for i in ICON_IDS if i.startswith("flag_")]

# Hand-written classroom sentences (icon ids in tap order).
SEED_SENTENCES = [
    "comm_good_morning teacher", "comm_hello teacher", "comm_hello comm_friend",
    "comm_goodbye teacher", "comm_goodbye comm_friend", "comm_good_morning comm_friend",
    "comm_excuse teacher question", "comm_excuse teacher help please",
    "teacher help please", "help please", "help please teacher", "teacher question",
    "question please", "teacher come_here please", "come_here please",
    "not_understand repeat please", "not_understand show_me please", "confused help please",
    "confused question teacher", "repeat please", "repeat comm_slower please",
    "comm_slower please", "comm_louder please", "comm_talk teacher", "show_me again please",
    "understand thankyou", "understand yes", "yes understand", "no not_understand",
    "more_time please", "wait please", "wait more_time please", "done teacher",
    "finished teacher", "done worksheet", "finished worksheet teacher", "done read",
    "comm_i_want water please", "comm_i_want food please", "water please",
    "water please thankyou", "food please", "food please thankyou",
    "comm_i_want rest please", "rest please", "need_break please", "comm_i_want need_break please",
    "tired need_break please", "tired rest please", "go_outside please",
    "comm_i_want go_outside please", "cold teacher", "hot water please", "hot go_outside please",
    "too_loud stop please", "too_loud comm_talk teacher", "pencil please", "give_me pencil please",
    "comm_i_want pencil please", "tissue please", "give_me tissue please", "sick tissue please",
    "fix_chair please", "fix_chair teacher help", "bag please", "give_me bag please",
    "open_it bag please", "open_it please help", "give_me notebook please", "notebook please",
    "give_me worksheet please", "worksheet please", "computer please", "comm_i_want computer please",
    "comm_i_like computer", "comm_i_like read", "comm_i_like activity", "comm_i_like group_work",
    "comm_i_dont_like group_work", "comm_i_dont_like activity", "comm_i_like comm_friend",
    "happy thankyou", "happy teacher", "happy comm_friend", "excited activity",
    "excited go_outside", "proud done", "proud math_answer", "calm thankyou",
    "sad teacher", "sad help please", "sad comm_talk teacher", "scared teacher help",
    "scared help please", "worried question teacher", "worried comm_talk teacher",
    "angry stop", "angry need_break please", "frustrated help please", "frustrated need_break",
    "frustrated not_understand help", "embarrassed comm_talk teacher", "bored activity please",
    "bored comm_i_want computer", "sick teacher", "sick rest please", "sick help teacher",
    "tired teacher", "stop please", "stop no", "no thankyou", "yes please", "yes thankyou",
    "thankyou teacher", "comm_sorry teacher", "comm_sorry comm_friend", "comm_goodjob comm_friend",
    "comm_myturn please", "comm_yourturn", "comm_myturn read", "comm_myturn try_it",
    "try_it again", "try_it please", "again please", "more_a please", "more_a water please",
    "give_me more_a please", "comm_agree", "comm_agree yes", "comm_disagree no",
    "comm_listen please", "comm_listen teacher", "sit_down please", "line_up go_a",
    "go_a go_outside", "group_work comm_friend", "comm_i_want group_work comm_friend",
    "write notebook", "read again", "read please", "write please help", "activity done",
    "question math_answer", "math_answer num_5", "math_answer yes", "math_answer question",
    "math_numbers help", "math_count help please", "math_shapes question",
    "math_measure help", "math_time question", "math_money question", "math_pattern help",
    "math_bigger math_smaller question", "math_more math_less question", "math_hundred",
    "not_understand math_numbers", "help math_count please", "show_me math_shapes please",
    # People
    "people_me sick", "people_me tired", "people_me happy", "people_me sad", "people_me done",
    "people_me comm_i_want water please", "people_me need_break please",
    "comm_i_want people_mom", "comm_i_want people_dad", "comm_i_want people_mom please",
    "people_mom come_here please", "comm_talk people_mom please", "comm_talk people_dad please",
    "sick people_nurse please", "comm_i_want people_nurse please", "people_me sick people_nurse",
    "go_a places_office people_nurse", "people_teacher help please", "people_teacher question",
    "comm_hello people_teacher", "comm_good_morning people_teacher", "thankyou people_teacher",
    "comm_hello people_principal", "comm_talk people_principal", "people_aide help please",
    "people_aide come_here please", "comm_hello people_friend", "comm_i_like people_friend",
    "group_work people_classmate", "comm_sorry people_classmate", "people_classmate stop please",
    "comm_myturn people_classmate", "comm_goodjob people_classmate", "happy people_friend",
    # Places
    "go_a places_bathroom please", "comm_i_want go_a places_bathroom please",
    "go_a places_cafeteria", "food places_cafeteria", "comm_i_want go_a places_cafeteria",
    "go_a places_library please", "read places_library", "comm_i_like places_library",
    "go_a places_gym", "excited places_gym", "comm_i_like places_gym",
    "go_a places_playground please", "go_outside places_playground", "happy places_playground",
    "go_a places_office please", "go_a places_classroom", "line_up go_a places_hallway",
    "go_a places_home", "comm_i_want go_a places_home", "sick go_a places_home please",
    "tired go_a places_home", "people_mom places_home", "places_classroom too_loud",
    # Not understanding
    "not_understand repeat", "not_understand show_me", "not_understand help please",
    "not_understand comm_slower please", "people_me not_understand repeat please",
    "not_understand question teacher", "not_understand repeat comm_slower please",
]


def _country_sentences() -> list[list[str]]:
    """One sentence per flag so every country is a known token. Kept unweighted so
    flags don't crowd out the everyday continuations of these frames."""
    frames = [["comm_i_like"], ["comm_i_want", "go_a"], ["question"], ["people_me"]]
    return [frames[i % len(frames)] + [flag] for i, flag in enumerate(FLAG_IDS)]


def _math_sentences() -> list[list[str]]:
    """Every arithmetic fact with operands 0-10 and a result in 0-10:
    num op num equals num (every tenth one starts with math_answer)."""
    ops = {"math_plus": lambda a, b: a + b, "math_minus": lambda a, b: a - b,
           "math_times": lambda a, b: a * b, "math_divide": lambda a, b: a // b}
    out = []
    for op, f in ops.items():
        for a in range(11):
            for b in range(1, 11):
                r = f(a, b)
                if not 0 <= r <= 10 or (op == "math_divide" and a % b):
                    continue
                seq = [f"num_{a}", op, f"num_{b}", "math_equals", f"num_{r}"]
                out.append(["math_answer"] + seq if len(out) % 10 == 0 else seq)
    return out


def seed_corpus() -> list[list[str]]:
    return [s.split() for s in SEED_SENTENCES] * 4 + _country_sentences() + _math_sentences()


def db_sentences() -> list[list[str]]:
    """Sentences from aac_logs.icon_sequence (one JSON list per sent message)."""
    from sqlalchemy import create_engine, text

    url = os.getenv("DATABASE_URL", "sqlite:///./vocalink.db")
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    try:
        with create_engine(url).connect() as conn:
            rows = conn.execute(text(
                "SELECT icon_sequence FROM aac_logs WHERE icon_sequence IS NOT NULL"
            )).fetchall()
    except Exception as e:
        print(f"Could not read aac_logs.icon_sequence ({e.__class__.__name__}); using seed data only")
        return []
    known = set(ICON_IDS)
    out = []
    for (raw,) in rows:
        try:
            seq = [t for t in json.loads(raw) if t in known]
        except (TypeError, ValueError):
            continue
        if seq:
            out.append(seq)
    return out


# ── Evaluation ────────────────────────────────────────────────────────────────
# Every predictor maps a prefix (icons tapped so far) to a ranked list of
# candidate next icons. Like NextIconLSTM.predict, baselines never suggest the
# icon that was just tapped, so all models are scored under the same rule.

SLICES = {
    "all":          lambda seq, t: True,
    "context = 1":  lambda seq, t: t == 1,
    "context >= 2": lambda seq, t: t >= 2,
    "math":         lambda seq, t: _is_math(seq),
    "math answer":  lambda seq, t: seq[t - 1] == "math_equals",
    "non-math":     lambda seq, t: not _is_math(seq),
}


def _is_math(seq):
    return any(tok.startswith(("num_", "math_plus", "math_minus", "math_times", "math_divide"))
               for tok in seq)


def _evaluate(predict, sentences):
    """Top-1 / top-3 next-icon accuracy at every position after the first icon,
    reported overall and per slice. Returns {slice: {top1, top3, n}}."""
    counts = {name: [0, 0, 0] for name in SLICES}
    for seq in sentences:
        for t in range(1, len(seq)):
            ranked = predict(seq[:t])
            for name, keep in SLICES.items():
                if keep(seq, t):
                    c = counts[name]
                    c[0] += ranked[:1] == [seq[t]]
                    c[1] += seq[t] in ranked[:3]
                    c[2] += 1
    return {name: {"top1": h1 / n if n else None, "top3": h3 / n if n else None, "n": n}
            for name, (h1, h3, n) in counts.items()}


def _perplexity(model, sentences):
    """exp(mean negative log-likelihood) of the true next icon, same positions as _evaluate."""
    nll, n = 0.0, 0
    for seq in sentences:
        for t in range(1, len(seq)):
            probs = model.next_probs(seq[:t])
            nll -= math.log(probs[model.index[seq[t]]] + 1e-12)
            n += 1
    return math.exp(nll / max(n, 1))


def _unigram_baseline(train):
    ranked = [w for w, _ in Counter(t for s in train for t in s).most_common()]
    return lambda prefix: [w for w in ranked if w != prefix[-1]][:3]


def _bigram_baseline(train):
    """Most frequent follower of the last icon, backing off to overall frequency."""
    counts = defaultdict(Counter)
    for seq in train:
        for a, b in zip(seq, seq[1:]):
            counts[a][b] += 1
    fallback = [w for w, _ in Counter(t for s in train for t in s).most_common()]

    def predict(prefix):
        last = prefix[-1]
        out = [w for w, _ in counts[last].most_common() if w != last]
        out += [w for w in fallback if w != last and w not in out]
        return out[:3]
    return predict


def _split(corpus, seed, test_frac):
    """Hold out a fraction of the *distinct* sentences. Training keeps the corpus
    weighting (repeats) of the remaining ones, as the final model does."""
    unique = sorted({tuple(s) for s in corpus})
    random.Random(seed).shuffle(unique)
    test = set(unique[:round(len(unique) * test_frac)])
    return [s for s in corpus if tuple(s) not in test], [list(s) for s in sorted(test)]


def _mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    m = sum(values) / len(values)
    return m, math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


SAMPLE_PREFIXES = [
    ["water"], ["sick"], ["tired"], ["confused"], ["not_understand"], ["comm_i_want"],
    ["comm_i_like"], ["people_me"], ["people_mom"], ["go_a"], ["sick", "go_a"],
    ["line_up", "go_a"], ["teacher"], ["happy", "teacher"], ["sick", "teacher"],
    ["num_3", "math_plus", "num_4", "math_equals"], ["num_8", "math_minus", "num_2", "math_equals"],
]


def _write_report(path_base, report):
    with open(path_base + ".json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    cfg, data, ev = report["config"], report["data"], report["evaluation"]
    pct = lambda v: "–" if v is None else f"{v * 100:.1f}%"
    pm = lambda s: f"{pct(s['mean'])} ± {s['std'] * 100:.1f}" if s["mean"] is not None else "–"
    lines = [
        "# Next-icon LSTM — training results", "",
        f"Generated {report['generated']} by `train_lstm.py`.", "",
        "## Data", "",
        "| | |", "|---|---|",
        f"| Icons in vocabulary | {data['vocab_size']} |",
        f"| Real student sentences (aac_logs) | {data['real_sentences']} |",
        f"| Seed sentences (hand-written + generated) | {data['seed_sentences']} |",
        f"| Training sequences (after weighting) | {data['training_sequences']} |",
        f"| Distinct sentences | {data['distinct_sentences']} |",
        f"| Average sentence length | {data['avg_length']:.2f} icons |", "",
        "## Model and training", "",
        "| | |", "|---|---|",
        f"| Architecture | Embedding ({cfg['embed_dim']}) → LSTM ({cfg['hidden_dim']}) → softmax ({data['vocab_size']}) |",
        f"| Parameters | {cfg['parameters']:,} |",
        f"| Optimizer | Adam, lr {cfg['lr']}, weight decay {cfg['weight_decay']}, grad clip {cfg['clip']} |",
        f"| Epochs | {cfg['epochs']} |",
        f"| Training time (final model) | {report['final_model']['train_seconds']:.1f} s |",
        f"| Weights file size | {report['final_model']['weights_kb']:.0f} KB |", "",
        "## Held-out evaluation", "",
        f"{ev['test_frac']:.0%} of distinct sentences held out, repeated over {len(ev['seeds'])} random "
        f"splits (seeds {', '.join(map(str, ev['seeds']))}); mean ± standard deviation. "
        "Every position after the first icon is scored; the icon just tapped is never suggested.", "",
        "| Slice | Positions | Model | Top-1 | Top-3 |", "|---|---|---|---|---|",
    ]
    for name in SLICES:
        n = ev["summary"]["lstm"][name]["n"]
        for model in ("lstm", "bigram", "unigram"):
            s = ev["summary"][model][name]
            lines.append(f"| {name} | {n:.0f} | {model} | {pm(s['top1'])} | {pm(s['top3'])} |")
    ppl = ev["summary"]["lstm_perplexity"]
    lines += ["", f"LSTM held-out perplexity: {ppl['mean']:.2f} ± {ppl['std']:.2f}", "",
              "## Training loss (final model)", "", "| Epoch | Loss |", "|---|---|"]
    curve = report["final_model"]["loss_curve"]
    for e in sorted({1, *range(5, len(curve) + 1, 5), len(curve)}):
        lines.append(f"| {e} | {curve[e - 1]:.4f} |")
    lines += ["", "## Sample predictions (final model)", "", "| Icons tapped | Top-3 next icons |", "|---|---|"]
    for row in report["final_model"]["samples"]:
        preds = ", ".join(f"{w} ({p * 100:.0f}%)" for w, p in row["predictions"])
        lines.append(f"| {' → '.join(row['prefix'])} | {preds} |")
    with open(path_base + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(here, "lstm_weights.npz"))
    ap.add_argument("--report", default=os.path.join(here, "results", "training_results"),
                    help="writes <report>.json and <report>.md")
    ap.add_argument("--eval-seeds", type=int, default=5, help="number of random held-out splits")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--no-db", action="store_true")
    args = ap.parse_args()

    real = [] if args.no_db else db_sentences()
    seed = seed_corpus()
    corpus = seed + real * REAL_WEIGHT
    print(f"{len(real)} real sentences from DB, {len(corpus)} training sequences total")

    # Held-out evaluation over several splits: LSTM vs. bigram vs. unigram.
    seeds = list(range(1, args.eval_seeds + 1))
    runs = []
    for s in seeds:
        train_set, test_set = _split(corpus, s, args.test_frac)
        probe = NextIconLSTM(ICON_IDS)
        probe.train(train_set, epochs=args.epochs, verbose=False)
        run = {
            "seed": s, "train_sequences": len(train_set), "test_sentences": len(test_set),
            "lstm": _evaluate(lambda p: [w for w, _ in probe.predict(p, 3)], test_set),
            "bigram": _evaluate(_bigram_baseline(train_set), test_set),
            "unigram": _evaluate(_unigram_baseline(train_set), test_set),
            "lstm_perplexity": _perplexity(probe, test_set),
        }
        runs.append(run)
        print(f"split {s}: top-3  LSTM {run['lstm']['all']['top3']:.1%}  "
              f"bigram {run['bigram']['all']['top3']:.1%}  unigram {run['unigram']['all']['top3']:.1%}")

    summary = {}
    for model in ("lstm", "bigram", "unigram"):
        summary[model] = {}
        for name in SLICES:
            entry = {"n": sum(r[model][name]["n"] for r in runs) / len(runs)}
            for metric in ("top1", "top3"):
                m, sd = _mean_std([r[model][name][metric] for r in runs])
                entry[metric] = {"mean": m, "std": sd}
            summary[model][name] = entry
    m, sd = _mean_std([r["lstm_perplexity"] for r in runs])
    summary["lstm_perplexity"] = {"mean": m, "std": sd}
    for model in ("lstm", "bigram", "unigram"):
        a = summary[model]["all"]
        print(f"held-out mean  {model:8s} top-1 {a['top1']['mean']:.1%}  top-3 {a['top3']['mean']:.1%}")

    # Final model on everything.
    model = NextIconLSTM(ICON_IDS)
    start = time.perf_counter()
    curve = model.train(corpus, epochs=args.epochs)
    train_seconds = time.perf_counter() - start
    model.save(args.out)
    print(f"saved {args.out}")

    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    distinct = {tuple(s) for s in corpus}
    report = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "config": {"epochs": args.epochs, "lr": 0.01, "weight_decay": 1e-4, "clip": 5.0,
                   "embed_dim": model.D, "hidden_dim": model.H,
                   "parameters": int(sum(v.size for v in model.params.values()))},
        "data": {"vocab_size": len(model.vocab) - 1, "real_sentences": len(real),
                 "seed_sentences": len(seed), "training_sequences": len(corpus),
                 "distinct_sentences": len(distinct),
                 "avg_length": sum(len(s) for s in distinct) / len(distinct)},
        "evaluation": {"test_frac": args.test_frac, "seeds": seeds, "runs": runs, "summary": summary},
        "final_model": {
            "train_seconds": train_seconds, "loss_curve": curve,
            "weights_kb": os.path.getsize(args.out if args.out.endswith(".npz") else args.out + ".npz") / 1024,
            "samples": [{"prefix": p, "predictions": model.predict(p, 3)} for p in SAMPLE_PREFIXES],
        },
    }
    _write_report(args.report, report)
    print(f"wrote {args.report}.md and {args.report}.json")


if __name__ == "__main__":
    main()
