# Double DQN and replay audit

The former learner used the target network's maximum next-state value. The corrected update selects the next action with the online network and evaluates that action with the target network:

`y = r + gamma * (1 - done) * Q_target(s_next, argmax_a Q_online(s_next, a))`

The current Q-value comes from the online network at the executed action. The update uses Huber loss, Adam at 0.001, discount 0.99, and gradient norm clipping at 10. The target starts synchronized and copies the online weights every 100 optimizer updates. Replay is a 10,000-entry FIFO with uniform samples of 64 transitions without replacement. The trainer attempts one update per environment action and begins updating when replay reaches 64 entries. The expected update count is max(0, steps − 63). Goal, collision, and timeout all end the replay transition under the established finite-horizon convention.

The existing remember mock already had the current signature. Regression tests distinguish standard DQN from Double DQN using disagreeing networks and check executed state transitions and penalty isolation. Training summaries now record environment steps, optimizer updates, the learner type, and candidate starts. See [test_results.txt](test_results.txt) for the current suite output. Existing weight-only checkpoints remain loadable and retain their original training provenance.
