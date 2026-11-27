# Related work and how FABR differs

> Team: check every citation against the original paper before submission. These notes describe
> how the work relates to ours; they are not quotations. In particular, confirm the exact
> wording of Alshiekh et al.'s two post-posed update options (the shielding section of the
> paper) before relying on the mapping below.

## What existing shielding work does

**Shielding (Alshiekh et al., 2018, "Safe Reinforcement Learning via Shielding", AAAI).**
A shield, computed from a formal safety specification, monitors the learner. *Preemptive*
shielding removes unsafe actions before the agent chooses. *Post-posed* shielding lets the agent
propose an action and overrides it with a safe one if it is unsafe. The shield guarantees safety
during learning. Our shield is a post-posed shield of the simplest kind: exact, hand-coded,
deterministic.

For post-posed shielding the paper discusses how the learner should be updated when the
shield overrides a proposal. Our feedback modes implement the two options:

| Our mode | Alshiekh et al. | What is stored for the blocked proposal a |
|---|---|---|
| `cf` | option 1 | a punished update for a: (s, a, −10, terminal), in addition to the executed transition. In our deterministic grid this is exactly what a would have done, so we label it an **oracle**: it uses the outcome of an action that was never executed |
| `credit` | option 2 | the executed outcome credited to a: (s, a, r, s'), with r and s' from the replacement, instead of the executed transition |

Neither option is about *which* safe action to substitute. Under `credit`, however, the choice
matters, because the replacement's outcome becomes the value of the blocked action. That is
where our v2 core tests the replacement rules.

**Probabilistic shields (Jansen et al., 2020, "Safe Reinforcement Learning Using Probabilistic
Shields", CONCUR).** These extend shielding to stochastic models by allowing actions whose
estimated risk is below a threshold. Again, the focus is on deciding safety.

**ADVICE (Bethell, Gerasimou, Calinescu & Imrie, "Safe Reinforcement Learning in Black-Box
Environments via Adaptive Shielding", arXiv:2405.18180; ECAI 2025).** ADVICE *learns* a
post-shield without handcrafted safety knowledge:

- a contrastive autoencoder learns a latent space that separates safe from unsafe state–action
  pairs;
- a nearest-neighbour classifier in that space flags unsafe actions;
- the shield's caution adapts over time.

The authors report substantially fewer safety violations than prior safe-exploration methods at
competitive reward, in continuous state and action spaces.

**Action masking and invalid-action handling** (e.g. Huang & Ontañón, 2022, "A Closer Look at
Invalid Action Masking in Policy Gradient Algorithms") removes forbidden actions from the
policy's choice set. With masking the agent never proposes an unsafe action, so there is no
replacement decision, and a masked policy is not trained to avoid unsafe actions without the mask.

## Large-margin losses: DQfD

**DQfD (Hester et al., 2018, "Deep Q-learning from Demonstrations", AAAI).** DQfD adds a
supervised large-margin classification loss to the TD loss on demonstration data:

    J_E(Q) = max_a [Q(s, a) + l(a_E, a)] − Q(s, a_E)

Here l is a positive margin for a ≠ a_E. This forces the demonstrated action a_E to be valued
at least a margin above every other action. FABR v2 borrows this mechanism and turns it around.
There is no demonstrator. Instead, the shield tells us that a_prop is *not* acceptable in s, so
the loss pushes Q(s, a_prop) at least m below the best safe action:

    relu(Q(s, a_prop) − max_{a ∈ safe(s)} Q(s, a) + m)

Like DQfD's loss, ours is zero once the margin holds. The `fabr_v2_fixedtarget` ablation
replaces the hinge with a squared loss that never switches off. It tests whether that property
matters.

## How FABR differs

| | Shielding / ADVICE | FABR v1 | FABR v2 |
|---|---|---|---|
| Question answered | *Is* this action unsafe? | *Which* safe action replaces it? | What should the agent *learn* from the block? |
| Safety knowledge | Formal spec (shielding) or learned classifier (ADVICE) | Takes any shield as given | Takes any shield as given; uses only "blocked", not the blocked action's outcome |
| Effect on learning | Update option 1 (punish) or 2 (credit), or unspecified | Replacement adapted to the agent's unsafe-proposal rate | Margin loss on the blocked proposal; replacement is highest-Q |
| Goal after training | Shield usually remains deployed | Safe **without** the shield | Safe **without** the shield |

**ADVICE is compared conceptually, not reimplemented**, because:

- the problem ADVICE solves (learning the shield) does not arise with our exact shield;
- a comparison against our exact shield would mix shield accuracy with what the agent learns
  from the shield;
- ADVICE targets continuous control. Re-implementing its contrastive autoencoder for a 10×10
  grid would test our port, not the method.

ADVICE and FABR are complementary: a learned shield such as ADVICE could supply the safe-action
set and the "blocked" signal that FABR v2's margin loss uses. Unlike `cf`, the margin loss does
not need to know what the blocked action would have done, which a learned shield cannot provide.

The experimental baselines are therefore the conventional replacement rules (random safe,
highest-Q, highest-Q with a tuned intervention penalty, a feedback-driven mix, FABR v1). Each is
combined with the two Alshiekh update options (`cf`, `credit`) or no feedback (`none`), and the
unshielded learner is the reference.

## Supporting background

- **Double DQN:** van Hasselt, Guez & Silver (2016), "Deep Reinforcement Learning with Double
  Q-learning", AAAI.
- **DQN:** Mnih et al. (2015), "Human-level control through deep reinforcement learning",
  Nature.
- **n-step returns:** Sutton & Barto (2018), *Reinforcement Learning: An Introduction*, 2nd ed.,
  ch. 7. They are used in Rainbow (Hessel et al., 2018, AAAI).
- **Potential-based reward shaping** preserves the optimal policy: Ng, Harada & Russell (1999),
  "Policy invariance under reward transformations", ICML.
- **Boltzmann (softmax) action selection:** Sutton & Barto (2018).
- **Survey:** García & Fernández (2015), "A Comprehensive Survey on Safe Reinforcement
  Learning", JMLR.
