# Job B shared-error audit

These are the test300 chats that **all four strong arms get wrong**: Jev + examples, the 500- and 2,000-chat adapters,
and the 500-chat adapter + examples. That's **37 of 300**. The first 30 by id were reviewed.

**Reviewer:** a single AI-assisted pass, not blind to the gold label. Treat this as indicative,
not as a measured inter-annotator agreement.

| Verdict | Count | Chats |
|---|---|---|
| **Gold looks wrong**: the action comes after the chat ended, repeats a step already done, or contradicts the playbook the models followed | 10 | 167, 183, 190, 295, 961, 1970, 2070, 4164, 4730, 4886 |
| **The models' answer is also reasonable**: more than one valid next step | 15 | 337, 960, 1341, 1540, 1711, 2206, 2599, 2914, 3707, 5468, 6292, 6382, 6585, 6673, 7148 |
| **Genuine model error** | 5 | 2879, 3169, 3180, 5845, 7066 |

**25 of 30 (83%) are label problems, not model failures.** Scaled to the 37 shared errors, about 31 of 300 chats (about 10%)
can't fairly be matched against the single recorded label. That suggests a ceiling **near 0.90**, not 0.71–0.73. Each strong
method gets about 27% of chats wrong, but only 37 chats are wrong for all four, so most errors aren't shared. **About 10 points
of job B error are label problems. The 0.71–0.73 plateau isn't fully explained by them.**

The 15 "also reasonable" cases are the softer category. A blind second look is set up in `results/jobB_blind_review.md`.

Examples:
- **183, 190:** the customer has said goodbye, and the gold "next action" is `verify-identity`. That's role-play noise.
- **1970, 2070:** the FAQ was already searched (`[Agent action] Searching the FAQ pages`), and the gold repeats `search-faq` or `search-policy`.
  The models pick the next step instead.
- **4164:** a mystery-fee query with account ID and order ID given. The playbook (Order Issue / Status Mystery Fee) says
  `verify-identity`, which all four models chose. The gold is `validate-purchase`.
- **6292, 7148:** adding an item to an order. The gold is `make-purchase`, the models say `update-order`. Both fit the playbook.
- **3169 (model error):** the shipping status was just recorded as in transit, and the next step is `offer-refund`. The models re-check shipping or update the order.

Raw extract: chat tails and every arm's prediction, from results/runs.csv.

## Blind second review (10 of the 15 "also reasonable" cases)

A separate AI-assisted reviewer, with no prior context, given only `jobB_blind_review.md` and told not to open the key, judged each of two candidates, the gold label and the models' majority answer in random order.
Unblinded against `jobB_blind_key.json`:

| Result | Count | Chats |
|---|---|---|
| Only the **models'** answer acceptable (gold wrong) | 6 | 337, 1711, 2206, 3707, 5468, 6673 |
| Both acceptable | 3 | 960, 1341, 6292 |
| Only the gold acceptable | 1 | 2599 |
| Neither | 0 | |

In **9 of 10** the models' answer was acceptable, and in 6 the blind reviewer preferred it to the gold. The "also reasonable"
category held up: if anything, it understated how often the gold label is the problem. **The estimate stands: about 10 points
of job B error are label problems** (ceiling near 0.90), and the 0.71–0.73 plateau is still not explained by them. Caveat:
the blind reviewer is also an AI model, not a human annotator.
