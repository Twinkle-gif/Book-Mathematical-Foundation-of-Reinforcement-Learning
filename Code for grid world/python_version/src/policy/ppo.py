# ppo.py
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import wandb
from examples.arguments import args
from src.envs.grid_world import GridWorld

class PolicyNetwork(nn.Module):
    """
    策略网络（Actor）
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)

class ValueNetwork(nn.Module):
    """
    价值网络（Critic）
    """
    def __init__(self, input_size, hidden_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PPOAgent:
    """
    PPO Agent实现
    """
    def __init__(self, state_size, action_size, 
                 learning_rate=3e-4, 
                 discount_factor=0.99,
                 epsilon_clip=0.2,
                 epochs=10,
                 buffer_size=2048):
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon_clip = epsilon_clip
        self.epochs = epochs
        
        # 设备选择
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 策略网络和价值网络
        self.policy_network = PolicyNetwork(state_size, 64, action_size).to(self.device)
        self.value_network = ValueNetwork(state_size, 64).to(self.device)
        
        # 优化器
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)
        
        # 经验缓冲区
        self.buffer = []
        self.buffer_size = buffer_size
        
    def act(self, state):
        """
        根据当前策略选择动作
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # 获取动作概率分布
        action_probs = self.policy_network(state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        
        # 采样动作
        action = action_dist.sample()
        
        # 计算动作的对数概率
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob.item()
    
    def evaluate(self, state, action):
        """
        评估给定状态和动作的价值
        """
        # 策略评估
        action_probs = self.policy_network(state)
        action_dist = torch.distributions.Categorical(action_probs)
        log_probs = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        
        # 价值评估
        state_values = self.value_network(state).squeeze()
        
        return log_probs, state_values, entropy
    
    def remember(self, state, action, reward, next_state, done, log_prob):
        """
        存储经验
        """
        self.buffer.append((state, action, reward, next_state, done, log_prob))
    
    def compute_advantages(self, rewards, values, dones):
        """
        计算优势函数
        """
        advantages = []
        gae = 0
        values = values + [0]  # 添加最后一个状态的价值为0
        
        for i in reversed(range(len(rewards))):
            if dones[i]:
                delta = rewards[i] - values[i]
                gae = delta
            else:
                delta = rewards[i] + self.discount_factor * values[i+1] - values[i]
                gae = delta + self.discount_factor * 0.95 * gae  # GAE参数设为0.95
            
            advantages.insert(0, gae)
        
        return advantages
    
    def update(self):
        """
        更新策略和价值网络
        """
        if len(self.buffer) < self.buffer_size:
            return
            
        # 提取缓冲区数据
        states = torch.FloatTensor([e[0] for e in self.buffer]).to(self.device)
        actions = torch.LongTensor([e[1] for e in self.buffer]).to(self.device)
        rewards = [e[2] for e in self.buffer]
        next_states = torch.FloatTensor([e[3] for e in self.buffer]).to(self.device)
        dones = [e[4] for e in self.buffer]
        old_log_probs = torch.FloatTensor([e[5] for e in self.buffer]).to(self.device)
        
        # 计算状态价值
        values = self.value_network(states).squeeze().detach().cpu().numpy().tolist()
        
        # 计算优势函数
        advantages = self.compute_advantages(rewards, values, dones)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # 计算回报（return）
        returns = advantages + torch.FloatTensor(values[:-1]).to(self.device)  # 移除最后一个0值
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # 标准化
        
        # 多轮更新
        for _ in range(self.epochs):
            # 评估当前策略
            log_probs, state_values, entropy = self.evaluate(states, actions)
            
            # 计算比率
            ratios = torch.exp(log_probs - old_log_probs)
            
            # 计算PPO损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 价值函数损失
            value_loss = F.mse_loss(state_values, returns)
            
            # 熵正则化
            entropy_loss = entropy.mean()
            
            # 更新策略网络
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            # 更新价值网络
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
        
        # 清空缓冲区
        self.buffer = []
    
    def get_policy_matrix(self, env):
        """
        获取策略矩阵用于可视化
        """
        policy_matrix = []
        for i in range(env.num_states):
            x = i % env.env_size[0]
            y = i // env.env_size[0]
            state = (x, y)
            
            # 编码状态
            state_encoded = state_to_coordinates(state, env.env_size)
            state_tensor = torch.FloatTensor(state_encoded).unsqueeze(0).to(self.device)
            
            # 获取动作概率
            action_probs = self.policy_network(state_tensor).detach().cpu().numpy()[0]
            policy_matrix.append(action_probs)
            
        return policy_matrix

def state_to_coordinates(state, env_size):
    """
    将状态转换为坐标表示
    """
    state_idx = state[1] * env_size[0] + state[0]
    x = state[0] / (env_size[0] - 1) if env_size[0] > 1 else 0
    y = state[1] / (env_size[1] - 1) if env_size[1] > 1 else 0
    return np.array([x, y])

def train_ppo_wandb(env: GridWorld, 
                    num_episodes=1000,
                    use_wandb=True, 
                    project_name="grid-world-ppo"):
    """
    训练PPO算法并使用wandb记录数据
    
    Parameters:
    - env: GridWorld环境
    - num_episodes: 训练回合数
    - use_wandb: 是否使用wandb记录
    - project_name: wandb项目名称
    """
    # 初始化wandb
    if use_wandb:
        wandb.init(
            project=project_name,
            config={
                "num_episodes": num_episodes,
                "learning_rate": 3e-4,
                "discount_factor": 0.99,
                "epsilon_clip": 0.2,
                "epochs": 10,
                "buffer_size": 2048,
                "env_size": env.env_size,
            }
        )
    
    # 确定状态大小
    state_size = 2  # 使用坐标表示
    action_size = len(env.action_space)
    
    ppo_agent = PPOAgent(state_size, action_size)
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = state_to_coordinates(state, env.env_size)
        
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            # 选择动作
            action, log_prob = ppo_agent.act(state)
            
            # 执行动作
            next_state, reward, done, _ = env.step(env.action_space[action])
            next_state = state_to_coordinates(next_state, env.env_size)
            
            # 存储经验
            ppo_agent.remember(state, action, reward, next_state, done, log_prob)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # 定期更新网络
            if len(ppo_agent.buffer) >= ppo_agent.buffer_size:
                ppo_agent.update()
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # 计算滑动平均奖励
        if len(episode_rewards) >= 100:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
        else:
            avg_reward = np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths)
        
        # 记录到wandb
        if use_wandb:
            wandb.log({
                "episode_reward": total_reward,
                "episode_length": steps,
                "average_reward": avg_reward,
                "average_length": avg_length,
                "episode": episode
            })
        
        # 每100回合打印一次平均回报
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{num_episodes}, Average Reward: {avg_reward:.2f}")
    
    # 结束wandb运行
    if use_wandb:
        wandb.finish()
    
    return ppo_agent, episode_rewards

# 使用示例
if __name__ == "__main__":
    # 创建环境
    env = GridWorld()
    
    # 训练PPO算法并使用wandb记录
    print("Training PPO with wandb logging...")
    try:
        ppo_agent, rewards = train_ppo_wandb(
            env, 
            num_episodes=1000,
            use_wandb=False,  # 设置为True启用wandb
            project_name="grid-world-ppo"
        )
    except Exception as e:
        print(f"无法使用wandb，错误信息: {e}")
        print("使用普通训练模式...")
        ppo_agent, rewards = train_ppo_wandb(
            env, 
            num_episodes=500,
            use_wandb=False
        )
    
    # 可视化最终学到的策略
    env.reset()
    env.render()
    policy_matrix = ppo_agent.get_policy_matrix(env)
    env.add_policy(policy_matrix)
    
    print("Training completed. Showing learned policy.")
    input("Press Enter to continue...")