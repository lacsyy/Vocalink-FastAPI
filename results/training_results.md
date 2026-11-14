# Next-icon LSTM — training results

Generated 2026-11-14T19:07:45 by `train_lstm.py`.

## Data

| | |
|---|---|
| Icons in vocabulary | 147 |
| Real student sentences (aac_logs) | 0 |
| Seed sentences (hand-written + generated) | 1100 |
| Training sequences (after weighting) | 1100 |
| Distinct sentences | 431 |
| Average sentence length | 3.55 icons |

## Model and training

| | |
|---|---|
| Architecture | Embedding (24) → LSTM (48) → softmax (147) |
| Parameters | 24,820 |
| Optimizer | Adam, lr 0.01, weight decay 0.0001, grad clip 5.0 |
| Epochs | 40 |
| Training time (final model) | 43.7 s |
| Weights file size | 188 KB |

## Held-out evaluation

15% of distinct sentences held out, repeated over 5 random splits (seeds 1, 2, 3, 4, 5); mean ± standard deviation. Every position after the first icon is scored; the icon just tapped is never suggested.

| Slice | Positions | Model | Top-1 | Top-3 |
|---|---|---|---|---|
| all | 160 | lstm | 36.0% ± 2.0 | 59.1% ± 3.0 |
| all | 160 | bigram | 32.2% ± 1.8 | 53.2% ± 3.4 |
| all | 160 | unigram | 10.0% ± 1.5 | 30.1% ± 2.9 |
| context = 1 | 65 | lstm | 17.9% ± 4.4 | 42.9% ± 2.6 |
| context = 1 | 65 | bigram | 14.2% ± 2.4 | 42.3% ± 5.8 |
| context = 1 | 65 | unigram | 9.2% ± 2.5 | 13.9% ± 3.0 |
| context >= 2 | 95 | lstm | 48.4% ± 3.2 | 70.1% ± 5.7 |
| context >= 2 | 95 | bigram | 44.5% ± 1.2 | 60.7% ± 2.7 |
| context >= 2 | 95 | unigram | 10.6% ± 3.3 | 41.1% ± 3.7 |
| math | 107 | lstm | 41.9% ± 2.8 | 72.1% ± 4.7 |
| math | 107 | bigram | 33.8% ± 2.0 | 60.3% ± 3.6 |
| math | 107 | unigram | 0.0% ± 0.0 | 26.0% ± 2.6 |
| math answer | 26 | lstm | 27.8% ± 8.7 | 71.6% ± 16.3 |
| math answer | 26 | bigram | 20.6% ± 6.2 | 29.1% ± 9.4 |
| math answer | 26 | unigram | 0.0% ± 0.0 | 7.5% ± 10.5 |
| non-math | 53 | lstm | 24.3% ± 3.2 | 33.2% ± 3.0 |
| non-math | 53 | bigram | 28.9% ± 2.1 | 39.0% ± 2.4 |
| non-math | 53 | unigram | 29.9% ± 4.1 | 38.3% ± 4.8 |

LSTM held-out perplexity: 17.73 ± 2.44

## Training loss (final model)

| Epoch | Loss |
|---|---|
| 1 | 3.3645 |
| 5 | 2.4498 |
| 10 | 2.4079 |
| 15 | 2.3529 |
| 20 | 2.3482 |
| 25 | 2.3324 |
| 30 | 2.3479 |
| 35 | 2.3148 |
| 40 | 2.2974 |

## Sample predictions (final model)

| Icons tapped | Top-3 next icons |
|---|---|
| water | please (99%), thankyou (0%), places_playground (0%) |
| sick | tissue (26%), rest (17%), help (14%) |
| tired | need_break (45%), rest (21%), go_a (18%) |
| confused | question (71%), help (27%), show_me (1%) |
| not_understand | show_me (42%), repeat (15%), question (14%) |
| comm_i_want | go_a (22%), people_mom (19%), food (13%) |
| comm_i_like | read (13%), places_gym (11%), activity (10%) |
| people_me | need_break (18%), sick (12%), comm_i_want (11%) |
| people_mom | places_home (68%), come_here (22%), help (3%) |
| go_a | places_gym (14%), places_classroom (12%), places_home (11%) |
| sick → go_a | places_home (98%), places_bathroom (0%), places_cafeteria (0%) |
| line_up → go_a | places_hallway (99%), please (0%), places_home (0%) |
| teacher | question (54%), help (28%), come_here (12%) |
| happy → teacher | come_here (42%), help (34%), places_home (16%) |
| sick → teacher | help (90%), question (7%), come_here (1%) |
| num_3 → math_plus → num_4 → math_equals | num_9 (33%), num_10 (24%), num_7 (16%) |
| num_8 → math_minus → num_2 → math_equals | num_7 (23%), num_9 (20%), num_6 (13%) |
