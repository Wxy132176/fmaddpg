# fmaddpg
                  Multidimensional Resource Load-aware Task Migration in Mobile Edge Computing 
# Abstract
 With the increasing demands for low latency, high computational power, and distributed execution in applications such as au
tonomous driving, augmented reality (AR), and smart manufacturing, Mobile Edge Computing (MEC) has emerged as a solution
 by bringing computation and storage closer to end users. In MEC environments, complex computational tasks are often struc
tured as workflows, consisting of multiple interdependent subtasks that require distributed execution across multiple MEC servers.
 However, workflow tasks in MEC environments often involve complex data and temporal dependencies, requiring frequent task
 migration across multiple MEC servers due to user mobility. Inefficient task migration can lead to increased execution delays,
 communication overhead, and server overload, ultimately degrading system performance and service quality. Therefore, how to
 effectively optimize workflow task migration strategies to ensure on-time task completion while achieving balanced resource al
location among edge servers remains a key challenge in workflow scheduling for MEC environments. To address the challenge,
 this paper proposes a comprehensive strategy integrating user trajectory prediction and federated deep reinforcement learning to
 jointly optimize workflow migration and resource allocation. A hybrid GRU-2LSTM model predicts user mobility trajectories,
 while a Federated Multi-Agent Deep Deterministic Policy Gradient (FMADDPG) algorithm optimizes task migration and resource
 allocation. Simulation results demonstrate that the proposed strategy reduces average load imbalance by 10%–20% and lowers
 workflow timeout rates by 7%–27%, highlighting its effectiveness in workflow scheduling, resource optimization, and overall sys
tem efficiency in dynamic MEC environments
# Dataset
Supported datasets:

Montage/Ligo/CyberShake

## Getting the Data

If you're looking to replicate our experiments, The workflows are generated using the generator provided by:
https://github.com/pegasus-isi/WorkflowGenerator/tree/master/bharathi

