from setuptools import setup, find_packages

setup(
    name="roboquest",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "mujoco>=3.1.0",
        "gymnasium>=0.29.0",
        "stable-baselines3>=2.3.0",
        "numpy>=1.24.0",
        "onnx>=1.20.0",
        "onnxruntime>=1.17.0",
        # ブラウザビューアー。0.9 系は API 破壊 + mjlab との mujoco pin 衝突のため固定。
        # mjswan の wheel は Requires-Python が <3.13 なので、Python 3.13 では
        # 依存解決から外す（中身は 3.13 でも動くので、Colab では setup_colab.sh /
        # ノートブックが --ignore-requires-python を付けて別途インストールする）。
        "mjswan==0.8.2; python_version < '3.13'",
        "mediapy>=1.1.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.66.0",
        "pandas>=2.0.0",
    ],
)
