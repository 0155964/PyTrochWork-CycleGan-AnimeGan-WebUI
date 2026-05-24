# PyTrochWork-CycleGan-AnimeGan-WebUI
streamlitCloud展示用
# CORE-ENGINE: 多模态图像数字处理与仿真平台

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![ONNX](https://img.shields.io/badge/ONNX_Runtime-Fast_Inference-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Cloud_Deployed-FF4B4B.svg)
![CycleGAN](https://img.shields.io/badge/CycleGAN-Image_Translation-orange.svg)

本平台是一个集成了多维度图像风格转换与物理级景深仿真的轻量化 Web 应用。基于深度学习网络（CycleGAN 与 AnimeGAN）自研训练，并进行了深度的工程化重构，剥离了笨重的 PyTorch 依赖，实现全平台秒级推理。

 🌐 **在线体验体验 (Live Demo)：** [点击此处访问 Streamlit Cloud 部署环境](https://pytrochwork-cyclegan-animegan-webui-auki9dj7dcpxla3nbrzqyd.streamlit.app)

---

## ✨ 核心特性 (Core Features)

- 🚀 **计算图静态化与异构加速**：将动态图（`.pt` / `.pth`）全量导出为 `ONNX` 静态计算图格式，摆脱了对独立显卡（GPU）和庞大深度学习框架的依赖，在普通 CPU 环境下即可实现数百毫秒级的极速推理。
- 🎨 **多模态矩阵支持**：
  - **风格迁移**：街景转动漫风 (AnimeGAN)
  - **实体转换**：苹果与橘子双向生成 (CycleGAN)
  - **艺术渲染**：现实照片与莫奈油画双向转换 (CycleGAN)
  - **物理仿真**：手机原图与单反大光圈景深双向转换 (CycleGAN)
- 🛠️ **自定义模型挂载**：支持用户在前端动态上传本地自训练的 `.onnx` 权重文件进行即时推理，并内置内存安全保护（推理完毕自动销毁释放）。
- 💻 **工业级响应式 UI**：突破原生 Streamlit 界面限制，注入自定义 CSS 与 DOM 监听，实现侧边栏自适应平铺块与沉浸式工作流。

---

## 🏗️ 架构与技术栈 (Tech Stack)

- **前端架构**：`Streamlit` + 自定义 CSS 注入
- **推理引擎**：`ONNX Runtime` (CPUExecutionProvider)
- **图像处理**：`Pillow (PIL)` + `NumPy`
- **底层算法（训练端）**：`PyTorch`、`CycleGAN` (ResNet-9blocks)、`AnimeGAN`

---

## 📂 核心目录结构 (Directory Structure)

```text
├── app.py                  # 平台前端主入口与路由控制
├── requirements.txt        # 云端部署精简依赖清单
├── export_onnx.py          # (开发端) PyTorch -> ONNX 转换脚本
├── onnx_models/            # 静态计算图权重存储目录 (核心资产)
│   ├── anime_style.onnx
│   ├── apple_to_orange.onnx
│   ├── orange_to_apple.onnx
│   ├── photo_to_monet.onnx
│   ├── monet_to_photo.onnx
│   ├── iphone_to_dslr.onnx
│   └── dslr_to_iphone.onnx
└── README.md               # 项目说明文档
```

*(注：为保证云端轻量化与部署效率，已剔除原始数据集 `datasets/` 与 PyTorch 动态权重 `checkpoints/`，所有模型均以更高效的 `.onnx` 格式呈现。)*

---

## 🚀 本地部署指南 (Local Deployment)

如果您希望在本地节点运行本系统，请按照以下步骤操作：

**1. 克隆仓库**
```bash
git clone https://github.com/0155964/PyTrochWork-CycleGan-AnimeGan-WebUI.git
cd PyTrochWork-CycleGan-AnimeGan-WebUI
```

**2. 安装轻量级依赖**
```bash
pip install -r requirements.txt
```

**3. 启动节点服务**
```bash
streamlit run app.py
```
服务启动后，平台将在您的本地浏览器 `http://localhost:8501` 就绪。

---
**© 2026 多模态图像数字处理与仿真平台** | Designed & Developed for AIGC Engineering