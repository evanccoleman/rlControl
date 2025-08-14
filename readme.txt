GOAL:

1. Does memory (LSTM PER) improve agent performance in continuous control
environments like the Mujoco walkers?

2. Do agents perform better in POMDP with memory (LSTM PER)?

=================================

WHERE OLIVIA LEFT OFF:

1. POMDP agents can be created but not loaded

2. Tuned hyperparameter settings for ant agents can be found in paramFiles/
(they were optimized for either 10,000 trials or 3 days).

3. Hyperparameter settings for POMDP ant agents are being optimized
on the cluster, and ant agents with tuned hyperparameters are being
trained on the cluster too (Olivia's account).

4. Must find good hyperparameters for hopper and halfcheetah envs
and then train those tuned agents on the cluster. Next, load those trained
agents and test their performance.

5. Optionally, get POMDP agents to load somehow (see KNOWN BUGS section).

=================================

SUMMARY OF FILES:

walkers_v1.py creates and loads agents into envs using hyperparameters
passed in from the command line.

optimizers_v2.py uses Optuna to find the best hyperparameters. 

walkers_v2.py creates and loads agents using hyperparameters found by 
the optimizer (takes files of hyperparameters from paramFiles/)

statemaskingwrapper.py artificially creates a POMDP by removing the
specified number of indices from the returned observation.

custom_ddpg.py is Olivia's attempt at writing a DDPG agent. It compiles in walkers_v1.py, but it does not work well.

getgraphs.py takes a file with trained-tuned agents (from test_agents/),
runs them each in the same type of env, and then creates plots of their
performance (discounted rewards).

=================================

SUMMARY OF DIRECTORIES:

agents_{}/ each hold agents that have been trained for x-many steps on
Olivia's pc

graphs/ is where graphs from getgraphs.py are found

paramFiles/ holds tuned hyperparameters for agents

test_agents/ holds files naming trained agents that should be loaded into
getgraphs.py and have their performance evaluated

tuned_agents_ant/ stores agents that have been trained using best-tuned
hyperparameters

NOTE: directories like graphs/, paramFiles/, and tuned_agents_ant/
have some output files that were solely created to test that programs
generated correct output and saved the output in the correct place (which
they do). These outputs are not necessarily finished or final products
to show off.

=================================

KNOWN BUGS:

1. Due to error checking in stablebaselines, a POMDP tuned agent that has 
been saved to a zip file cannot be loaded because of incongruent sizes of
observation spaces. It's weird because the env that was saved and the env
that gets passed in should have the same observation space size, and they
should both have a StateMaskingWrapper. I don't know what went wrong with
the loading process. (NOTE: a fully observable MDP tuned agent can still
be loaded to walkers_v2 or optimizers_v2.py)

=================================

OTHER RESOURCES TO REFER TO OUTSIDE THE REPO: 
- LaTeX memory project (ppoForDummies)
- Olivia's notes on Word and Apple Notes (on her local device)
