"""几何变换。

职责:
    - resize / crop / flip / rotate；必须同步变换 seg mask、bbox、以及距离标签的有效性
    - 相机内参矩阵要跟着一起变（单目测距依赖内参），这是最容易出错的地方
"""
