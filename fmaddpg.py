import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import math

# 环境参数
NE = 10  # 边缘节点数量
NU = 8  # 用户数量
Rs = 50  # 边缘节点服务半径(m)
B = 300  # 总带宽(Mbps)
F = 32  # 每个边缘节点的计算资源(GHz)
Me = 32  # 每个边缘节点的内存资源(GB)
tau = 0.5  # 时隙大小(s)
deadlines = [110, 500, 1800]  # 工作流截止时间(s)
PH = 100  # 超时惩罚系数

# FMADDPG算法参数
gamma = 0.9  # 折扣因子
phi = 0.8  # 软更新系数
batch_size = 64  # 批处理大小
actor_lr = 0.001  # Actor学习率
critic_lr = 0.001  # Critic学习率
memory_size = 1200  # 经验回放池大小
target_update_interval = 16  # 目标网络更新间隔

class Task:
    """任务模型"""
    def __init__(self, task_id, workflow_id, input_data, need_cycles, output_data):
        self.task_id = task_id  # 任务ID
        self.workflow_id = workflow_id  # 所属工作流ID
        self.input_data = input_data  # 输入数据大小(GB)
        self.need_cycles = need_cycles  # 所需计算资源(GB)
        self.output_data = output_data  # 输出数据大小(GB)
        self.dependencies = []  # 依赖的任务ID列表
        self.status = 0  # 0:等待 1:执行 2:迁移
        self.location = -1  # 当前所在边缘节点ID

class Workflow:
    """工作流模型"""
    def __init__(self, workflow_id, user_id, deadline):
        self.workflow_id = workflow_id  # 工作流ID
        self.user_id = user_id  # 所属用户ID
        self.deadline = deadline  # 截止时间
        self.tasks = []  # 任务列表
        self.start_time = -1  # 开始时隙
        self.end_time = -1  # 结束时隙
        self.ifdone = 0  # 0:执行中 1:已完成

    def add_task(self, task):
        self.tasks.append(task)

    def add_dependency(self, task_id, depend_id):
        # 为任务添加依赖关系
        for task in self.tasks:
            if task.task_id == task_id:
                task.dependencies.append(depend_id)
                break

class EdgeNode:
    """边缘节点模型"""
    def __init__(self, node_id):
        self.node_id = node_id  # 节点ID
        self.cpu_remaining = F  # 剩余计算资源
        self.mem_remaining = Me  # 剩余内存资源
        self.tasks = []  # 当前节点上的任务
        self.datasets = []  # 当前节点上的数据集
        self.covered_users = set()  # 覆盖的用户

class MECEnvironment:
    """MEC环境模型"""
    def __init__(self):
        self.edge_nodes = [EdgeNode(i) for i in range(NE)]  # 边缘节点列表

        # 用户位置要自行生成或者用现成的数据集
        self.users = []

         # 工作流初始化这里需要对三种工作流进行预处理后送进来
        self.workflows = []

        self.current_time = 0  # 当前时隙
        self.init_workflows()  # 初始化工作流任务

    def init_workflows(self):

        for wf in self.workflows:
            task_num = random.randint(5, 15)
            for i in range(task_num):
                input_data = random.uniform(0.5, 2.0)
                need_cycles = random.uniform(1.0, 5.0)
                output_data = input_data * random.uniform(0.3, 0.8)
                task = Task(i, wf.workflow_id, input_data, need_cycles, output_data)
                if i > 0:
                    task.dependencies.append(random.randint(0, i-1))
                wf.add_task(task)
            wf.start_time = self.current_time

    def get_node_load(self, node_id):

        node = self.edge_nodes[node_id]
        cpu_load = (F - node.cpu_remaining) / F
        mem_load = (Me - node.mem_remaining) / Me
        return cpu_load, mem_load

    def process_migrations(self, node_id, task_migrations, data_migrations):

        node = self.edge_nodes[node_id]

        for task_id, target_node in task_migrations:
            for task in node.tasks:
                if task.task_id == task_id:

                    task.status = 2
                    target = self.edge_nodes[target_node]
                    target.tasks.append(task)
                    node.tasks.remove(task)
                    task.location = target_node
                    task.status = 0
                    break

        for data_id, target_node in data_migrations:
            for data in node.datasets:
                if data["id"] == data_id:
                    data["status"] = 2
                    target = self.edge_nodes[target_node]
                    target.datasets.append(data)
                    node.datasets.remove(data)
                    data["status"] = 1
                    break

    def execute_tasks(self):

        for node in self.edge_nodes:
            for task in node.tasks:
                if task.status == 0 and self.check_dependencies(task):

                    if node.cpu_remaining >= task.need_cycles and node.mem_remaining >= task.input_data:
                        task.status = 1
                        node.cpu_remaining -= task.need_cycles
                        node.mem_remaining -= task.input_data

                        node.cpu_remaining += task.need_cycles
                        node.datasets.append({
                            "id": f"{task.workflow_id}_{task.task_id}",
                            "size": task.output_data,
                            "status": 1
                        })
                        task.status = 3

    def check_dependencies(self, task):

        wf = self.workflows[task.workflow_id]
        for depend_id in task.dependencies:
            depend_task = next(t for t in wf.tasks if t.task_id == depend_id)
            if depend_task.status != 3:
                return False
        return True

    def check_workflow_completion(self):

        for wf in self.workflows:
            if wf.ifdone == 1:
                continue
            all_done = all(t.status == 3 for t in wf.tasks)
            if all_done:
                wf.ifdone = 1
                wf.end_time = self.current_time

    def calculate_reward(self):

        total_load_imbalance = 0.0
        # 计算所有节点的负载失衡度
        cpu_loads = []
        mem_loads = []
        for node in self.edge_nodes:
            cpu, mem = self.get_node_load(node.node_id)
            cpu_loads.append(cpu)
            mem_loads.append(mem)
        avg_cpu = np.mean(cpu_loads)
        avg_mem = np.mean(mem_loads)

        # 计算系统负载失衡度
        for i in range(NE):
            cpu_diff = (cpu_loads[i] - avg_cpu) **2
            mem_diff = (mem_loads[i] - avg_mem)** 2
            total_load_imbalance += math.sqrt(cpu_diff) + math.sqrt(mem_diff)
        total_load_imbalance /= NE

        # 计算超时惩罚
        timeout_punish = 0
        for wf in self.workflows:
            if wf.ifdone == 0:
                current_delay = (self.current_time - wf.start_time) * tau
                if current_delay > wf.deadline:
                    timeout_punish += 1

        # 计算奖励
        reward = 1 / (total_load_imbalance + PH * timeout_punish + 1e-6)
        return reward, total_load_imbalance

    def get_observations(self):
        # 获取每个边缘节点的观测
        observations = []
        for node in self.edge_nodes:
            cpu_load, mem_load = self.get_node_load(node.node_id)
            # 构建观测向量
            obs = [
                cpu_load, mem_load,
                len(node.tasks), len(node.datasets),
                node.cpu_remaining / F, node.mem_remaining / Me,
                len(node.covered_users)
            ]
            observations.append(obs)
        return observations

    def step(self, actions):
        # 执行智能体动作，更新环境状态
        self.current_time += 1

        for node_id in range(NE):
            action = actions[node_id]

            task_migrations = [(int(action[0]*10), int(action[1]*NE))]
            data_migrations = [(int(action[2]*10), int(action[3]*NE))]
            self.process_migrations(node_id, task_migrations, data_migrations)

        # 2. 执行任务计算
        self.execute_tasks()

        # 3. 计算奖励与负载失衡度
        reward, load_imbalance = self.calculate_reward()

        # 4. 更新用户覆盖关系（模拟用户移动）
        for i in range(NU):
            # 随机移动用户
            self.users[i]["location"] += np.random.normal(0, 2, 2)
            # 更新覆盖关系
            for node in self.edge_nodes:
                if i in node.covered_users:
                    node.covered_users.remove(i)
            # 查找最近的边缘节点
            min_dist = float('inf')
            nearest_node = -1
            for node in self.edge_nodes:
                # 简化节点位置为ID*100
                node_pos = np.array([node.node_id * 100, 0])
                dist = np.linalg.norm(self.users[i]["location"] - node_pos)
                if dist < Rs and dist < min_dist:
                    min_dist = dist
                    nearest_node = node.node_id
            if nearest_node != -1:
                self.edge_nodes[nearest_node].covered_users.add(i)

        return self.get_observations(), reward, load_imbalance

class Actor(nn.Module):
    """Actor网络"""
    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()  # 输出动作范围[-1,1]

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x

class Critic(nn.Module):
    """Critic网络"""
    def __init__(self, obs_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class FMADDPGAgent:
    """FMADDPG智能体"""
    def __init__(self, node_id, obs_dim, action_dim):
        self.node_id = node_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        # 初始化网络
        self.actor = Actor(obs_dim, action_dim)
        self.actor_target = Actor(obs_dim, action_dim)
        self.critic = Critic(obs_dim, action_dim)
        self.critic_target = Critic(obs_dim, action_dim)
        # 优化器
        self.actor_optim = optim.RMSprop(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)
        # 经验回放池
        self.memory = deque(maxlen=memory_size)
        # 初始化目标网络参数
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target, source):
        # 软更新
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - phi) + param.data * phi)

    def select_action(self, obs):
        # 根据观测选择动作
        obs = torch.FloatTensor(obs).unsqueeze(0)
        action = self.actor(obs).detach().numpy()[0]
        return action

    def store_transition(self, obs, action, reward, next_obs):
        self.memory.append((obs, action, reward, next_obs))

    def train(self, agents):
        # 从经验池采样
        if len(self.memory) < batch_size:
            return
        samples = random.sample(self.memory, batch_size)
        obs_batch = torch.FloatTensor([s[0] for s in samples])
        action_batch = torch.FloatTensor([s[1] for s in samples])
        reward_batch = torch.FloatTensor([s[2] for s in samples]).unsqueeze(1)
        next_obs_batch = torch.FloatTensor([s[3] for s in samples])

        # 训练Critic
        with torch.no_grad():
            # 获取所有智能体的目标动作
            next_actions = []
            for i in range(NE):
                next_actions.append(agents[i].actor_target(next_obs_batch))
            next_actions = torch.cat(next_actions, dim=1)
            q_target = reward_batch + gamma * self.critic_target(next_obs_batch, next_actions)
        q_pred = self.critic(obs_batch, action_batch)
        critic_loss = nn.MSELoss()(q_pred, q_target)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # 训练Actor
        actor_loss = -self.critic(obs_batch, self.actor(obs_batch)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # 软更新目标网络
        if self.node_id % target_update_interval == 0:
            self.soft_update(self.actor_target, self.actor)
            self.soft_update(self.critic_target, self.critic)

def federated_aggregation(agents):
    """联邦学习参数聚合"""
    # 聚合所有Critic网络参数
    global_params = []
    # 初始化全局参数
    for param in agents[0].critic.parameters():
        global_params.append(param.data.clone() * (1.0/NE))

    # 累加其他智能体参数
    for i in range(1, NE):
        for idx, param in enumerate(agents[i].critic.parameters()):
            global_params[idx] += param.data.clone() * (1.0/NE)

    # 分发全局参数
    for agent in agents:
        for idx, param in enumerate(agent.critic.parameters()):
            param.data.copy_(global_params[idx])

# 主训练流程
if __name__ == "__main__":
    # 初始化环境与智能体
    env = MECEnvironment()
    obs_dim = 7  # 观测维度
    action_dim = 6  # 动作维度
    agents = [FMADDPGAgent(i, obs_dim, action_dim) for i in range(NE)]

    # 训练参数
    episodes = 5000
    max_steps = 100

    for episode in range(episodes):
        obs = env.get_observations()
        total_reward = 0.0
        for step in range(max_steps):
            # 智能体选择动作
            actions = [agents[i].select_action(obs[i]) for i in range(NE)]
            # 执行动作
            next_obs, reward, load_imbalance = env.step(actions)
            # 存储经验
            for i in range(NE):
                agents[i].store_transition(obs[i], actions[i], reward, next_obs[i])
            # 训练智能体
            for i in range(NE):
                agents[i].train(agents)
            # 联邦聚合
            if step % 10 == 0:
                federated_aggregation(agents)
            # 更新状态
            obs = next_obs
            total_reward += reward

