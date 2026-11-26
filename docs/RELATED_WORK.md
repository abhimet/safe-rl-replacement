# Related work and how FABR differs

> Team: check every citation against the original paper before submission. These notes describe
> how the work relates to ours; they are not quotations.

## What existing shielding work does

**Shielding (Alshiekh et al., 2018, "Safe Reinforcement Learning via Shielding", AAAI).**
A shield, computed from a formal safety specification, monitors the learner. *Preemptive*
shielding removes unsafe actions before the agent chooses. *Post-posed* shielding lets the agent
propose an action and overrides it with a safe one if it is unsafe. The shield guarantees safety
during learning. The paper is about *which actions are safe*, not about *which safe action* to
substitute or what that substitution teaches the agent. Our shield is a post-posed shield of the
simplest kind: exact, hand-coded, deterministic.

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

## How FABR differs

| | Shielding / ADVICE | FABR |
|---|---|---|
| Question answered | *Is* this action unsafe? | *Which* safe action replaces it? |
| Safety knowledge | Formal spec (shielding) or learned classifier (ADVICE) | Takes any shield as given (here an exact one) |
| Effect on learning | Replacement is typically fixed or unspecified | Replacement is a learning signal, adapted to the agent's own unsafe-proposal rate |
| Goal after training | Shield usually remains deployed | Agent should be safe **without** the shield |

ADVICE and FABR are **complementary, not competing**. ADVICE could supply the safe-action set
that FABR chooses from. For that reason we compare with ADVICE in the literature and do not run it
as an experimental baseline:

- the problem ADVICE solves (learning the shield) does not arise with our exact shield;
- a comparison against our exact shield would mix shield accuracy with replacement policy.

The experimental baselines are therefore the conventional *replacement rules* used with
post-posed shields:

- random safe action;
- highest-Q safe action;
- highest-Q with an intervention penalty;
- a feedback-driven mix of random and highest-Q;
- the unshielded learner as a lower bound.

The contribution we test is narrow and specific: **adapting the exploration temperature of the
replacement choice to the agent's recent unsafe-proposal rate**. The `fabr_fixed` ablation
separates the effect of this feedback from the amount of exploration alone.

## Supporting background

- **Double DQN:** van Hasselt, Guez & Silver (2016), "Deep Reinforcement Learning with Double
  Q-learning", AAAI.
- **DQN:** Mnih et al. (2015), "Human-level control through deep reinforcement learning",
  Nature.
- **Potential-based reward shaping** preserves the optimal policy: Ng, Harada & Russell (1999),
  "Policy invariance under reward transformations", ICML.
- **Boltzmann (softmax) action selection:** Sutton & Barto (2018), *Reinforcement Learning: An
  Introduction*, 2nd ed.
- **Survey:** García & Fernández (2015), "A Comprehensive Survey on Safe Reinforcement
  Learning", JMLR.
