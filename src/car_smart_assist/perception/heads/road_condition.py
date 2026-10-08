"""旧 Stage 1 路况头占位，配置中已禁用。

当前天气现象与光照由 perception/weather.py 的独立属性模型处理。
输出顺序和阈值见 configs/model/weather_classifier.yaml；pipeline 将多标签结果
传入 PerceptionResult.weather_attributes / weather_probabilities，不复用旧单类别编码。
"""
