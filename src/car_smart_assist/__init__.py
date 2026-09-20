"""car_smart_assist —— 复杂路况智能辅助驾驶决策系统。

两阶段架构:
    Stage 1  perception/  ACDC 恶劣天气数据的多任务感知
             - 接管边界判断 (handover boundary)
             - 路况分类 (road condition)
             - 车辆检测 + 前车/来车距离 (detection + distance)
    Stage 2  advisory/    把 Stage 1 的结构化输出翻译成给司机的自然语言建议

本文件只放包级元信息与导出，不写业务逻辑。
"""
