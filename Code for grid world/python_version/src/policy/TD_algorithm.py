import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)
import numpy as np
import random
import wandb
from examples.arguments import args
from src.envs.grid_world import GridWorld

class TDAlgorithm:
    """
    Temporal Difference (TD) Learning Algorithm
    实现TD(0)算法用于策略评估和控制
    """

    def __init__(self, env: GridWorld, learning_rate=0.1, discount_factor=0.9, epsilon=0.1, sarsa_n_steps=None):
        self.env = env
        self.learning_rate = learning_rate/sarsa_n_steps if sarsa_n_steps else learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # 初始化状态价值函数
        self.state_values = np.zeros(env.num_states)
        
        # 初始化Q函数 (状态-动作价值函数)
        self.q_values = np.zeros((env.num_states, len(env.action_space)))
        
        # 将坐标状态转换为一维索引的函数
        self._state_to_index = lambda state: state[1] * env.env_size[0] + state[0]

    def get_action_epsilon_greedy(self, state):
        """
        使用ε-贪婪策略选择动作
        """
        state_idx = self._state_to_index(state)
        
        if random.random() < self.epsilon:
            # 随机选择动作
            return random.choice(range(len(self.env.action_space)))
        else:
            # 贪婪选择最优动作
            return np.argmax(self.q_values[state_idx])

    def update_value_td0(self, state, reward, next_state, done):
        """
        TD(0)算法更新状态价值
        V(S_t) ← V(S_t) - α[V(S_t)-( R_{t+1} + γV(S_{t+1}) )]
        """
        state_idx = self._state_to_index(state)
        next_state_idx = self._state_to_index(next_state) if not done else None
        
        # TD目标
        if done:
            td_target = reward
        else:
            td_target = reward + self.discount_factor * self.state_values[next_state_idx]
        
        # TD误差
        td_error = self.state_values[state_idx] - td_target
        
        # 更新状态价值
        self.state_values[state_idx] -= self.learning_rate * td_error

    def update_q_value_td0(self, state, action, reward, next_state, done):
        """
        TD(0)算法更新动作价值函数(Q函数), 本质上就是Q-Learning的更新
        Q(S_t, A_t) ← Q(S_t, A_t) - α[Q(S_t,A_t)- ( R_{t+1} + γ max_a Q(S_{t+1}, a) )]
        """
        state_idx = self._state_to_index(state)
        next_state_idx = self._state_to_index(next_state) if not done else None
        
        # TD目标
        if done:
            td_target = reward
        else:
            td_target = reward + self.discount_factor * np.max(self.q_values[next_state_idx])
        
        # TD误差
        td_error = self.q_values[state_idx, action] - td_target
        
        # 更新Q值
        self.q_values[state_idx, action] -= self.learning_rate * td_error

    def sarsa_update(self, state, action, rewards, final_state, final_action, done):
        """
        SARSA算法更新 (on-policy TD控制)
        n=1时:
            Q(S_t, A_t) ← Q(S_t, A_t) - α[Q(S_t, A_t) - ( R_{t+1} + γQ(S_{t+1}, A_{t+1}) )]
        n步时:
            Q(S_t, A_t) ← Q(S_t, A_t) - α[Q(S_t, A_t) - ( R_{t+1} + γ*R_{t+2} + γ^2*R_{t+3} + ... + γ^n * Q(S_{t+n}, A_{t+n}) )]
        """
            

        state_idx = self._state_to_index(state)
        final_state_idx = self._state_to_index(final_state)
      
        td_target = self.q_values[final_state_idx, final_action]
        for reward in reversed(rewards):
            td_target = self.discount_factor * td_target + reward
        
        # TD误差
        td_error = self.q_values[state_idx, action] - td_target
        
        # 更新Q值
        self.q_values[state_idx, action] -= self.learning_rate * td_error

    def get_policy_from_q_values(self):
        """
        从Q函数中提取确定性策略
        π(s) = argmax_a Q(s, a)
        """
        policy = []
        for state_idx in range(self.env.num_states):
            best_action = np.argmax(self.q_values[state_idx])
            policy.append(best_action)
        return policy

    def get_policy_matrix(self):
        """
        获取策略矩阵用于可视化
        """
        policy_matrix = []
        for state_idx in range(self.env.num_states):
            action_probs = np.zeros(len(self.env.action_space))
            best_action = np.argmax(self.q_values[state_idx])
            action_probs[best_action] = 1.0
            policy_matrix.append(action_probs)
        return policy_matrix


def train_td_algorithm_wandb(env: GridWorld, algorithm_type="q_learning", num_episodes=1000, 
                             use_wandb=True, project_name="grid-world-rl", sarsa_steps=None):
    """
    训练TD算法并使用wandb记录数据
    
    Parameters:
    - env: GridWorld环境
    - algorithm_type: 算法类型 ("td0", "sarsa", "q_learning")
    - num_episodes: 训练回合数
    - use_wandb: 是否使用wandb记录
    - project_name: wandb项目名称
    - sarsa_steps: SARSA算法的n步TD算法的n值 (仅当algorithm_type为"sarsa"时使用)
    """
    # 初始化wandb
    if use_wandb:
        wandb.init(
            project=project_name,
            config={
                "algorithm_type": algorithm_type,
                "num_episodes": num_episodes,
                "learning_rate": 0.1,
                "discount_factor": 0.9,
                "epsilon": 0.1,
                "env_size": env.env_size,
            }
        )
    
    td_agent = TDAlgorithm(env, sarsa_n_steps=sarsa_steps)
    
    episode_rewards = []
    episode_lengths = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            if algorithm_type == "sarsa":
                rewards = []
                action = td_agent.get_action_epsilon_greedy(state)
                state_history = [state]
                action_history = [action]
                for _ in range(sarsa_steps if sarsa_steps else 1):
                    action = td_agent.get_action_epsilon_greedy(state)
                    state, reward, done, _ = env.step(env.action_space[action])
                    state_history.append(state)
                    action_history.append(action)
                    total_reward += reward
                    steps += 1
                    rewards.append(reward)
                    if done: break
                td_agent.sarsa_update(state_history[0], action_history[0], rewards, state_history[-1], action_history[-1], done)
                state = state_history[1]
            else:
                action = td_agent.get_action_epsilon_greedy(state)
                # 执行动作
                next_state, reward, done, _ = env.step(env.action_space[action])
                total_reward += reward
                steps += 1
                
                # 根据算法类型更新价值函数
                if algorithm_type == "td0":
                    td_agent.update_value_td0(state, reward, next_state, done)
                # elif algorithm_type == "sarsa":
                #     next_action = td_agent.get_action_epsilon_greedy(next_state)
                #     td_agent.sarsa_update(state, action, reward, next_state, next_action, done)
                #     action = next_action  # 为下一步准备
                elif algorithm_type == "q_learning":
                    td_agent.update_q_value_td0(state, action, reward, next_state, done)
                
                state = next_state
        
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
    
    return td_agent, episode_rewards


# 使用示例
if __name__ == "__main__":
    # 创建环境
    env = GridWorld()
    
    # 训练Q-Learning算法并使用wandb记录
    print("Training TD-algorithm with wandb logging...")
    try:
        q_learning_agent, rewards = train_td_algorithm_wandb(
            env, 
            algorithm_type="sarsa", 
            num_episodes=100000,
            use_wandb=True,  # 设置为True启用wandb
            project_name="grid-world-td-learning",
            sarsa_steps=8
            )
    except Exception as e:
        print(f"无法使用wandb，错误信息: {e}")
        print("使用普通训练模式...")
        q_learning_agent, rewards = train_td_algorithm_wandb(
            env, 
            algorithm_type="q_learning", 
            num_episodes=500,
            use_wandb=False
        )
    
    # 可视化最终学到的策略
    env.reset()
    env.render()
    policy_matrix = q_learning_agent.get_policy_matrix()
    env.add_policy(policy_matrix)
    env.add_state_values(q_learning_agent.state_values)
    
    print("Training completed. Showing learned policy.")
    input("Press Enter to continue...")