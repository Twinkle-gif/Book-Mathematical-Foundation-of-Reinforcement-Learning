# dqn.py
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

class DQN(nn.Module):
    """
    深度Q网络
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    """
    DQN Agent实现
    """
    def __init__(self, state_size, action_size, 
                 learning_rate=0.001, 
                 discount_factor=0.9,
                 epsilon=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.995,
                 buffer_size=10000,
                 batch_size=32,
                 target_update_freq=100,
                 state_representation:str=None):
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.state_representation = state_representation
        
        # 设备选择
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 主网络和目标网络
        self.q_network = DQN(state_size, 64, action_size).to(self.device)
        self.target_network = DQN(state_size, 64, action_size).to(self.device)
        self.update_target_network()
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 经验回放缓冲区
        self.memory = deque(maxlen=buffer_size)
        
        # 将坐标状态转换为一维索引的函数
        self._state_to_index = None  # 在训练函数中设置
        
    def update_target_network(self):
        """
        更新目标网络
        """
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        """
        存储经验到回放缓冲区
        """
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        """
        根据当前策略选择动作
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.detach().cpu().numpy())
    
    def replay(self):
        """
        从经验回放中采样并训练
        """
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach() # max返回的是(values, indices)，[0]取values
        # 计算目标Q值，这里的~dones对应的数学公式是：
        # Target = R_{t+1} + γ * max_a Q(S_{t+1}, a)  if not done
        # Target = R_{t+1}                           if done
        #保持贝尔曼方程的正确性
        # 未结束的episode：Target = r + γ * max_a Q(s',a)
        # 结束的episode：Target = r（因为没有未来奖励）
        target_q_values = rewards + (self.discount_factor * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def decay_epsilon(self):
        """
        衰减epsilon
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def get_q_values(self, state):
        """
        获取给定状态的所有动作的Q值
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.detach().cpu().numpy()[0] # 下面举例解释了为什么要加[0]
        #原始状态: [0.2 0.4]
        # 添加批次维度后: tensor([[0.2000, 0.4000]])
        # 形状: torch.Size([1, 2])

        # 神经网络输出（带批次维度）: tensor([[ 1.2000,  2.5000,  0.8000,  3.1000, -0.5000]])
        # 形状: torch.Size([1, 5])

        # 转换为NumPy数组: [[ 1.2  2.5  0.8  3.1 -0.5]]
        # 形状: (1, 5)

        # 索引[0]后的结果: [ 1.2  2.5  0.8  3.1 -0.5]
        # 形状: (5,)
    def get_policy_matrix(self, env):
        """
        获取策略矩阵用于可视化
        """
        policy_matrix = []
        for i in range(env.num_states):
            x = i % env.env_size[0]
            y = i // env.env_size[0]
            state = (x, y)
            
            # 根据状态表示方法编码状态
            if hasattr(self, 'state_representation') and self.state_representation == "one_hot":
                state_encoded = state_to_one_hot(state, env.env_size)
            else:
                state_encoded = state_to_coordinates(state, env.env_size)
                
            q_values = self.get_q_values(state_encoded)
            action_probs = np.zeros(len(env.action_space))
            best_action = np.argmax(q_values)
            action_probs[best_action] = 1.0
            policy_matrix.append(action_probs)
        return policy_matrix

def state_to_one_hot(state, env_size):
    """
    将状态转换为one-hot编码
    """
    state_idx = state[1] * env_size[0] + state[0]
    one_hot = np.zeros(env_size[0] * env_size[1])
    one_hot[state_idx] = 1
    return one_hot

def state_to_coordinates(state, env_size):
    """
    将状态转换为坐标表示
    """
    state_idx = state[1] * env_size[0] + state[0]
    x = state[0] / (env_size[0] - 1) if env_size[0] > 1 else 0
    y = state[1] / (env_size[1] - 1) if env_size[1] > 1 else 0
    return np.array([x, y])

def train_dqn_wandb(env: GridWorld, 
                    state_representation="coordinates",
                    num_episodes=1000,
                    use_wandb=True, 
                    project_name="grid-world-dqn"):
    """
    训练DQN算法并使用wandb记录数据
    
    Parameters:
    - env: GridWorld环境
    - state_representation: 状态表示方法 ("one_hot" 或 "coordinates")
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
                "learning_rate": 0.001,
                "discount_factor": 0.9,
                "epsilon_decay": 0.995,
                "epsilon_min": 0.01,
                "buffer_size": 10000,
                "batch_size": 32,
                "state_representation": state_representation,
                "env_size": env.env_size,
            }
        )
    
    # 确定状态大小
    if state_representation == "one_hot":
        state_size = env.num_states
    else:  # coordinates
        state_size = 2
        
    action_size = len(env.action_space)
    dqn_agent = DQNAgent(state_size, action_size)
    # 保存状态表示方法以便后续使用
    dqn_agent.state_representation = state_representation
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        if state_representation == "one_hot":
            state = state_to_one_hot(state, env.env_size)
        else:
            state = state_to_coordinates(state, env.env_size)
            
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            action = dqn_agent.act(state)
            next_state, reward, done, _ = env.step(env.action_space[action])
            
            if state_representation == "one_hot":
                next_state_encoded = state_to_one_hot(next_state, env.env_size)
            else:
                next_state_encoded = state_to_coordinates(next_state, env.env_size)
                
            dqn_agent.remember(state, action, reward, next_state_encoded, done)
            
            state = next_state_encoded
            total_reward += reward
            steps += 1
            
            dqn_agent.replay()
            
            if done:
                dqn_agent.update_target_network()
        
        dqn_agent.decay_epsilon()
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
                "epsilon": dqn_agent.epsilon,
                "episode": episode
            })
        
        # 每100回合打印一次平均回报
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{num_episodes}, Average Reward: {avg_reward:.2f}, Epsilon: {dqn_agent.epsilon:.3f}")
    
    # 结束wandb运行
    if use_wandb:
        wandb.finish()
    
    return dqn_agent, episode_rewards

# 使用示例
if __name__ == "__main__":
    # 创建环境
    env = GridWorld()
    
    # 训练DQN算法并使用wandb记录
    print("Training DQN with wandb logging...")
    try:
        dqn_agent, rewards = train_dqn_wandb(
            env, 
            state_representation="coordinates",
            num_episodes=1000,
            use_wandb=False,  # 设置为True启用wandb
            project_name="grid-world-dqn"
        )
    except Exception as e:
        print(f"无法使用wandb，错误信息: {e}")
        print("使用普通训练模式...")
        dqn_agent, rewards = train_dqn_wandb(
            env, 
            state_representation="coordinates",
            num_episodes=500,
            use_wandb=False
        )
    
    # 可视化最终学到的策略
    env.reset()
    env.render()
    policy_matrix = dqn_agent.get_policy_matrix(env)
    env.add_policy(policy_matrix)
    
    print("Training completed. Showing learned policy.")
    input("Press Enter to continue...")