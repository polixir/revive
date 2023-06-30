import gym
import random
import numpy as np
import matplotlib.pyplot as plt

from gym import spaces


class PipelineEnv(gym.Env):
    """管道模拟环境
    
    管道流量控制模拟环境，用于模拟水流经过一条管道， 通过调节管道开始出的水闸控制水流的流量。管道长度为100米，为了避免复杂的流体运动计算，简化流速为固定流速5米/s,且管道不同部分的水流量不会产生影响。
    因此水流通过整个管道的时间固定为20s(时间步)。管道开始处的最大水流量为200。整个环境的运行时间步数为1000。
    管道末端的目标水流量受外部策略影响，目标流量会随机从[50,80,110,140]获得，每一时间步有0.003的概率会随机选择新的目标水流量，整条轨迹1000个时间步中平均会更改3次目标水流量。
    每次初始化环境，环境的末端输出水流量会存在随机的损失，用于模拟隐变量对系统的影响，每条轨迹的损失系数是固定的。
    观测状态：[管道开始处的水流量， 管道末端的水流量,  管道末端的目标水流量]
    动作空间：[对管道开始处的水流量调节值(+-1之间)，]
    奖励函数：
    奖励函数由两部分组成，一部分是管道末端的水流量和目标水流量之间的均方误差，用于引导控制策略控制管道末端的水流量；另外一部分是动作的均方误差，用于避免频繁进行流量调节。
    
    参数：
        target_flow_rate： 管道末尾的目标流量，用于模拟目标动态变化
        time_delay：随机选择倒数第几节管道进行输出，用于模拟不定长时延带来的影响
        tributary_rate：末端流量被支流分走的比例，用于模拟观测不到的因素对系统产生的影响
        obs_list:obs是否包括历史观测，默认False，不包含
        delta_target:管道末端输出的延迟，默认False，无影响
        include_action:观测是否包含历史动作，默认False，不包含
        obs_tributary：观测到的末端流量是否是损失之后的水流量，默认True
        
    
    """
    def __init__(self, 
                 target_flow_rate=100, 
                 time_delay=1, 
                 tributary_rate=0, 
                 obs_list=False, 
                 delta_target=False, 
                 include_action=False, 
                 obs_tributary=True,
                 seed=None):
        super(PipelineEnv, self).__init__() 
        self.seed(seed)
        # 环境参数
        self.channel_length = 100 # 管道长度
        self.max_flow_rate = 200 # 最大放水速度（立方米/秒）
        self.pipe_velocity = 5   # 流速（米/秒）
        
        # 目标流量（立方米/秒）
        self._target_flow_rate = target_flow_rate 
        self.target_flow_rate = None
        self.target_flow_rate_list = []
        
        self.delta_target = delta_target
        self.obs_tributary = obs_tributary
        
        # 时间延迟，关系到使用那一节管道作为输出
        self._time_delay = time_delay
        self.time_delay = None
        self.time_delay_list = []
        
        # 支流管道输出，定义输出时支流管道分走的流量比例
        self._tributary_rate = tributary_rate    
        self.tributary_rate = None
        self.tributary_rate_list = []
        
        # 状态空间和动作空间
        self.obs_list = obs_list
        self.include_action = include_action
        if self.obs_list:
            if self.include_action:
                obs_shape = (self.channel_length//self.pipe_velocity)*2 + 4
            else:
                obs_shape = self.channel_length//self.pipe_velocity + 2
        else:
            obs_shape = 3
        
        self.observation_space = spaces.Box(low=np.array([0 for i in range(obs_shape)]), high=np.array([self.max_flow_rate for i in range(obs_shape)])) # 水闸放水每秒流量
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32) # 水闸放水每秒流量调节值
        
        # 环境变量
        self.reset()
        self.seed(seed)
        
        # 初始化环境的其他组件
        
    def seed(self, seed=None):
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        random.seed(seed)
        
    def get_time_delay(self):
        if callable(self._time_delay):
            self.time_delay = self._time_delay(self.time_delay)
        else:
            self.time_delay = self._time_delay
    
        self.time_delay_list.insert(0, self.time_delay)
        
    def get_tributary_rate(self):
        if callable(self._tributary_rate):
            self.tributary_rate = self._tributary_rate(self.tributary_rate)
        else:
            self.tributary_rate = self._tributary_rate
            
        self.tributary_rate_list.insert(0, self.tributary_rate)
    
    def get_target_flow_rate(self):
        if callable(self._target_flow_rate):
            self.target_flow_rate = self._target_flow_rate(self.target_flow_rate)
        else:
            self.target_flow_rate = self._target_flow_rate
            
        self.target_flow_rate_list.insert(0, self.target_flow_rate)
        
    def step(self, action):
        self._step += 1
        self.get_target_flow_rate()
        self.get_tributary_rate()
        self.get_time_delay()
    
        # 对水闸防水每秒流量进行调节
        #action *= 10
        self.action_hisoty.insert(0,action.reshape(-1)[0])
        self.watergate_flow += np.clip(action, -10, 10)[0]
        
        # 防止水闸防水每秒流量小于0或大于最大放水速度
        self.watergate_flow = np.clip(self.watergate_flow, 0, self.max_flow_rate)
        
        # 更新action list
        self.action_list = [action[0], ] + self.action_list[:-1]

        # 更新管道list
        self.water_flow_list = [self.watergate_flow,] + self.water_flow_list[:-1]
        self.full_water_flow_list.insert(0, self.watergate_flow)
        
        self.downstream_flow = self.water_flow_list[-self.time_delay] * (1-self.tributary_rate)
        
        
        self.downstream_flow_list.insert(0, self.downstream_flow )
        #assert self.downstream_flow >= 0 and self.downstream_flow <= 200
        
        # 计算奖励
        reward = ((200 - abs(self.downstream_flow - self.target_flow_rate)) * 0.01) ** 2 
        reward -= ((action ** 2) * 0.01)[0]
        
        # 判断是否到达目标流量，如果到达则结束当前回合
        if self._step >= 1000:
            done = True
        else:
            done = False
        
        # 返回状态、奖励、是否结束、其他信息（空字典）
        if self.obs_list:
            if self.include_action:
                obs = np.array(tuple(self.action_list + self.water_flow_list[:-1] +[self.downstream_flow, self.target_flow_rate,]))
            else:
                obs = np.array(tuple(self.water_flow_list[:-1] +[self.downstream_flow, self.target_flow_rate,]))
        else:
            obs = np.array((self.water_flow_list[0], self.downstream_flow, self.target_flow_rate))
        
        if self.delta_target:
            obs[-1] = obs[-1] - obs[0]
            
        if not self.obs_tributary:
            obs[-2] = self.water_flow_list[-1]
            
        return (obs, reward, done, done, {})
    
    def reset(self, seed=None, options=None):
        self.seed(seed)
        self.reset_water_flow_list()
        self.target_flow_rate_list = []
        self.time_delay_list = []
        self.tributary_rate_list = []
        self.get_target_flow_rate()
        self.get_tributary_rate()
        self.get_time_delay()
        self._step = 0
        self.action_hisoty = [0,]
        
        # 水闸放水每秒流量
        self.watergate_flow = self.water_flow_list[0]
        # 末端每秒流量
        self.downstream_flow = self.water_flow_list[-self.time_delay] * (1- self.tributary_rate)
        
        self.downstream_flow_list = [self.downstream_flow,]
        # 返回初始化状态
        if self.obs_list:
            if self.include_action:
                obs = np.array(tuple(self.action_list + self.water_flow_list[:-1] +[self.downstream_flow, self.target_flow_rate,]))
            else:
                obs = np.array(tuple(self.water_flow_list[:-1] +[self.downstream_flow, self.target_flow_rate,]))
        else:
            obs =  np.array((self.water_flow_list[0], self.downstream_flow, self.target_flow_rate))
            
        if self.delta_target:
            obs[-1] = obs[-1] - obs[0]
        if not self.obs_tributary:
            obs[-2] = self.water_flow_list[-1]
            
        return obs, {}
        
    def reset_water_flow_list(self):
        lst = []
        for i in range(self.channel_length//self.pipe_velocity + 1):
            if i == 0:
                lst.append(random.uniform(80, 120))
            else:
                if random.random() < 1.1:
                    lst.append(lst[-1])
                else:
                    lst.append(random.uniform(80, 120))
        lst = lst[::-1]
        self.water_flow_list = lst
        self.full_water_flow_list = lst
        self.action_list = [0 for _ in self.full_water_flow_list] + [0, ]
    
    def render(self, mode='human'):
        fig, ax1 = plt.subplots(figsize=(10, 3))
        ax1.plot(range(len(self.target_flow_rate_list)), self.full_water_flow_list[:len(self.target_flow_rate_list)][::-1], label='Input Water Flow Rate')
        ax1.plot(range(len(self.target_flow_rate_list)), self.target_flow_rate_list[::-1], label='Target Flow Rate')
        ax1.plot(range(len(self.downstream_flow_list)), self.downstream_flow_list[::-1], label='Output Flow Rate')
        ax2 = ax1.twinx()
        ax2.plot(range(len(self.downstream_flow_list)), self.action_hisoty[:len(self.target_flow_rate_list)][::-1], label='Action', color='red')
        ax2.axhline(y=0, color='gray', linestyle='--', label="Zero line")
        ax2.set_ylim(-5, 5)
        ax2.set_ylabel('Action')
        # 添加图例并显示图像
        ax1.legend(loc='upper right')
        ax2.legend(loc='upper left')
        
        plt.show()

        
    def close(self):
        pass
    
def get_target_flow_rate(target_flow_rate=None):
    if random.random() < 0.003 or target_flow_rate is None:
        return random.choice([50,80,110,140])
    else:
        return target_flow_rate
    
def get_tributary_rate(tributary_rate=None):
    if random.random() < 0.003 or tributary_rate is None:
        return random.uniform(0.1,0.2)
    else:
        return tributary_rate
    
def get_tributary_rate_2(tributary_rate=None):
    if random.random() < -1 or tributary_rate is None:
        return random.uniform(0,0.2)
    else:
        return tributary_rate
    
def get_time_delay(time_delay=None):
    if random.random() < 0.01 or time_delay is None:
        return random.choice([1,2,3,4,5,6,7])
    else:
        return time_delay
    
def get_env(version = "flow-v4"):
    if version == "flow-v0":
        return PipelineEnv(get_target_flow_rate, 1, 0, include_action=False)
    elif version == "flow-v0-delta":
        return PipelineEnv(get_target_flow_rate, 1, 0, delta_target=True, include_action=False)
    elif version == "flow-v1":
        return PipelineEnv(get_target_flow_rate, 1, get_tributary_rate, include_action=False)
    elif version == "flow-v1-delta":
        return PipelineEnv(get_target_flow_rate, 1, get_tributary_rate, delta_target=True, include_action=False)
    elif version == "flow-v2":
        return PipelineEnv(get_target_flow_rate, 1, get_tributary_rate, include_action=True, obs_tributary=False)
    elif version == "flow-v3":
        return PipelineEnv(get_target_flow_rate, 1, get_tributary_rate, obs_list=False, include_action=False, obs_tributary=True)
    elif version == "flow-v4":
        return PipelineEnv(get_target_flow_rate, 1, get_tributary_rate_2, obs_list=False, include_action=False, obs_tributary=True)
    else:
        raise NotImplementedError