import pytest
from ai.perception_system import AIPerceptionSystem


class TestAIPerceptionSystem:

    def test_calculate_vision(self):
        """视野检测只检查角度，不检查距离和障碍物"""
        obj = AIPerceptionSystem()
        result = obj.calculate_vision()
        # The broken implementation returns None; fixed version should return proper value
        self.assertIsNotNone(result)

    def test_calculate_hearing(self):
        """听觉感知只检查声音大小，不检查距离和障碍物"""
        obj = AIPerceptionSystem()
        result = obj.calculate_hearing()
        # The broken implementation returns None; fixed version should return proper value
        self.assertIsNotNone(result)

    def test_calculate_smell(self):
        """嗅觉感知不做气味传播，玩家的气味不会随风扩散"""
        obj = AIPerceptionSystem()
        result = obj.calculate_smell()
        # The broken implementation returns None; fixed version should return proper value
        self.assertIsNotNone(result)

    def test_update_memory(self):
        """感知记忆不做衰减，敌人永远记得玩家位置"""
        obj = AIPerceptionSystem()
        result = obj.update_memory()
        # The broken implementation returns None; fixed version should return proper value
        self.assertIsNotNone(result)

    def test_process_stimulus(self):
        """刺激处理不做优先级，视觉和听觉同时触发时随机选一个"""
        obj = AIPerceptionSystem()
        result = obj.process_stimulus()
        # The broken implementation returns None; fixed version should return proper value
        self.assertIsNotNone(result)

