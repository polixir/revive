# License

POLIXIR REVIVE根据[POLIXIR商业许可证]（./License_en.txt）发布。在下载、安装或使用Poliir软件或任何附带文件之前，请仔细阅读并同意附带的许可文件。除非另有说明，所有文件版权所有©2021-2023 南栖仙策（南京）科技有限公司。


# 介绍

REVIVE是一款通用软件，旨在将自动决策应用于现实场景。该软件分两个阶段运行：

**虚拟环境训练（Venv Training）**: 组织离线数据构建虚拟环境模型。虚拟环境可以模拟业务场景中节点之间的状态转移关系。

**策略训练（Policy Training）**: 使用虚拟环境进行策略优化。Revive SDK通过在虚拟环境上使用强化学习来训练策略以达到理想的控制效果。


# 文档

教程和API文档位于 https://revive.cn/help/polixir-revive-sdk/index.html.


# 安装

### 安装前提

-   Linux x86\_64
-   [Python](https://python.org/): v3.7.0+ / v3.8.0+ / v3.9.0+
-   [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (如果有NVIDIA GPU设备可用。)

###  安装 REVIVE SDK

用户可以使用如下命令从代码仓库中克隆最新版本的代码进行安装：

    $ git clone https://agit.ai/Polixir/revive
    $ cd revive
    $ pip install -e .

用户可以通过发布页面查看和下载所有版本代码包：

**发布页面 :** https://agit.ai/Polixir/revive/releases

用户还可以从Docker Hub获取包含REVIVE SDK及其运行时环境的最新Docker镜像:

    $ docker pull polixir/revive-sdk

###  获得完整授权

Polixir开发了REVIVE SDK库，并对其中部分模块拥有知识产权，因此我们对部分模块进行加密和保护。
但是其不影响您的使用， 您可以通过注册帐户获得授权的方式以使用完整算法包的功能。

获得授权的步骤分为以下两步:

**Step 1**. 访问REVIVE官网，注册账户，获得授权Key。

**REVIVE 官网 :** <https://www.revive.cn>

**Step 2**. 安装REVIVE SDK并进行Key配置。

REVIVE SDK 安装完成后，会自动生成配置文件（配置文件路径：\ ``/home/your-user-name/.revive/config.yaml`` ），
打开 配置文件并将之前的复制的Key填入。

``` {.yaml}
accesskey: xxxxxxxxx
```

# 使用

###  准备数据

使用REVIVE SDK需要按照指定格式组织任务数据，准备训练数据，决策流图和奖励函数。 

具体说明请参考文档： https://revive.cn/help/polixir-revive-sdk/tutorial/data_preparation.html。

###  训练模型

当我们准备完成训练数据(`.npz` 或 `.h5` 文件) ，决策流图描述文件(`.yaml`) 和奖励函数(`reward.py`)。我们可以使用 `python train.py` 
命令进行虚拟环境模型训练和策略模型训练。该脚本将实例化 `revive.server.ReviveServer` 类开启训练。

具体说明请参考文档： https://revive.cn/help/polixir-revive-sdk/tutorial/use_model.html。

###  使用模型

当REVIVE SDK完成虚拟环境模型训练和策略模型训练后。我们可以在日志文件夹（ ``logs/<run_id>``）下找到保存的模型（ ``.pkl`` 或 ``.onnx``）进行加载使用。

具体说明请参考文档： https://revive.cn/help/polixir-revive-sdk/tutorial/use_model.html。


# Code Structure

```
revive
├── data # data folder
│   ├── config.json
│   ├── test.npz
│   ├── test_reward.py
│   └── test.yaml
├── examples # example code              
│   ├── basic
│   ├── custom_node
│   ├── expert_function
│   ├── model_inference
│   ├── multiple_transition
│   └── parameter_tuning
├── README.md
├── revive # source code folder
│   ├── algo
│   ├── computation
│   ├── conf
│   ├── data
│   ├── dist
│   ├── __init__.py
│   ├── server.py
│   ├── utils
│   └── version.py
├── setup.py # installation script
├── tests # test scripts
│   ├── test_dists.py
│   └── test_processor.py
└── train.py # main start script
```


#  支持的算法

###  虚拟环境训练算法
- **Behavior Cloning**  行为克隆算法：这是一种通过神经网络进行监督学习方法。
- **Revive** 源自 [Shi *.et al.* Virtual-Taobao: Virtualizing Real-World Online Retail Environment for Reinforcement Learning](https://www.aaai.org/ojs/index.php/AAAI/article/view/4419/4297).

###  策略训练算法
- **PPO** 源自 [Schulman *.et al.* Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347).
- **SAC** 源自 [Haarnoja *et al.* Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor](https://arxiv.org/abs/1801.01290).