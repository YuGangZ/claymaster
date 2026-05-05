from abc import ABC, abstractmethod
import numpy as np


class BaseController(ABC):
    """控制器抽象基类 - 解耦RL与具体实现"""

    def __init__(self, state_dim: int = 16, control_dim: int = 3):
        self.state_dim = state_dim
        self.control_dim = control_dim

    @abstractmethod
    def set_target(self, target_state: np.ndarray) -> None:
        """设置目标状态（14维）"""
        pass

    @abstractmethod
    def control(self, current_state: np.ndarray, contact_info: dict) -> np.ndarray:
        """
        执行控制计算
        Args:
            current_state: 14维当前形状
            contact_info: 接触信息字典
        Returns:
            3维控制指令 [vx, vy, vz]
        """
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_status(self) -> dict:
        pass

    @abstractmethod
    def set_sub_target(self, sub_target: np.ndarray) -> None:
        """设置子目标（由RL调用）"""
        pass

    @abstractmethod
    def get_current_target(self) -> np.ndarray:
        """获取当前跟踪的目标"""
        pass