import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import time
import os
import tempfile

# ---------- 页面配置 ----------
st.set_page_config(
    page_title="多模态图像数字处理与仿真平台",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- 路由与状态初始化 ----------
if "task" in st.query_params:
    try:
        st.session_state.task_index = int(st.query_params["task"])
    except:
        st.session_state.task_index = 0
else:
    st.session_state.task_index = 0

# ---------- Gemini 级平铺自适应画布 CSS 注入 ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        font-family: 'Inter', -apple-system, sans-serif !important;
        background-color: #F4F7FC !important;
    }

    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #E2E8F0 !important;
    }

    /* ================= 1. 平铺选择块与纯 CSS 图标 ================= */
    .flat-menu-container {
        display: flex; flex-direction: column; gap: 8px; width: 100%; margin-top: 10px;
    }
    .menu-item {
        display: flex; align-items: center; gap: 16px; padding: 12px 16px;
        background-color: transparent; border-radius: 6px; text-decoration: none !important;
        color: #4A5568 !important; transition: all 0.2s ease; border-left: 3px solid transparent;
    }
    .menu-item:hover { background-color: #F4F7FC; color: #1A5AC6 !important; }
    .menu-item.active { background-color: #EBF1FA; color: #0B409C !important; font-weight: 600; border-left: 3px solid #1A5AC6; }
    .menu-text { font-size: 0.9rem; line-height: 1.4; white-space: normal !important; word-break: break-word !important; }

    .css-icon { width: 18px; height: 18px; position: relative; flex-shrink: 0; display: flex; justify-content: center; align-items: center; }
    .icon-anime::before, .icon-anime::after { content: ""; position: absolute; width: 9px; height: 9px; border: 2px solid #1A5AC6; border-radius: 1px; }
    .icon-anime::before { top: 0; left: 0; }
    .icon-anime::after { bottom: 0; right: 0; background: #ffffff; }
    .active .icon-anime::after { background: #EBF1FA; }
    .icon-fruit { border: 2px solid #1A5AC6; width: 14px; height: 14px; border-radius: 50%; }
    .icon-fruit::before { content: ""; position: absolute; width: 4px; height: 4px; background: #0B409C; border-radius: 50%; }
    .icon-monet::before { content: ""; position: absolute; width: 14px; height: 2px; background: #1A5AC6; box-shadow: 0 4px 0 #1A5AC6, 0 8px 0 #1A5AC6; top: 2px; }
    .icon-dslr { border: 2px solid #1A5AC6; width: 16px; height: 11px; border-radius: 2px; }
    .icon-dslr::before { content: ""; position: absolute; width: 4px; height: 4px; border: 1px solid #0B409C; border-radius: 50%; }
    .icon-custom::before { content: ""; position: absolute; width: 9px; height: 9px; border-left: 2px solid #1A5AC6; border-bottom: 2px solid #1A5AC6; transform: rotate(-45deg); top: 2px; }

    /* ================= 2. 终极修复：紧凑图标列与原生按钮保护 ================= */
    
    /* 强行提权原生展开按钮，保护它不被遮挡，作为我们的主控开关 */
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"] {
        z-index: 99999 !important; 
    }

    .collapsed-rail {
        display: none;
        position: fixed;
        top: 0;
        left: 0;
        width: 64px; /* 侧边图标列宽度 */
        height: 100vh;
        background-color: #ffffff;
        border-right: 1px solid #E2E8F0;
        z-index: 99990; /* 层级位于原生按钮之下，主页面之上 */
        flex-direction: column;
        align-items: center;
        padding-top: 5.5rem; /* 精确避开左上角原生 >> 按钮的高度 */
        gap: 1.5rem;
        box-shadow: 2px 0 10px rgba(0,0,0,0.04);
    }
    
    .rail-item {
        display: flex; justify-content: center; align-items: center;
        width: 38px; height: 38px; border-radius: 8px; text-decoration: none !important;
        transition: all 0.2s;
    }
    .rail-item:hover { background-color: #F4F7FC; transform: scale(1.08); }
    .rail-item.active { background-color: #EBF1FA; border: 1px solid #1A5AC6; }

    /* 最强嗅探：监听全避开版本号，只要原生侧边栏切换为收缩状态 (aria-expanded="false")，立刻显示图标列 */
    [data-testid="stAppViewContainer"]:has([data-testid="stSidebar"][aria-expanded="false"]) .collapsed-rail {
        display: flex !important;
    }

    /* 当图标列弹出时，主动让主容器内容向右平移 64px，防止文字被图标列挡住 */
    [data-testid="stAppViewContainer"]:has([data-testid="stSidebar"][aria-expanded="false"]) .main .block-container {
        padding-left: 80px !important;
    }

    /* ================= 3. 基础工业版面组件 ================= */
    .platform-header {
        display: flex; justify-content: space-between; align-items: center;
        background: #ffffff; padding: 1.2rem 2rem; border-radius: 8px;
        box-shadow: 0 4px 12px rgba(26, 90, 198, 0.03); border-bottom: 3px solid #1A5AC6; margin-bottom: 1.5rem;
    }
    .platform-brand { display: flex; align-items: center; gap: 12px; }
    .platform-logo { background: #1A5AC6; color: white; padding: 6px 14px; border-radius: 4px; font-weight: 700; font-size: 1rem; }
    .platform-title { font-size: 1.4rem; font-weight: 600; color: #0B409C; }

    section[data-testid="stFileUploader"] {
        border: 2px dashed #CBD5E1 !important; border-radius: 8px !important;
        padding: 1.5rem !important; background-color: #ffffff !important;
    }
    .stButton>button {
        background: linear-gradient(135deg, #1A5AC6, #0B409C) !important; color: white !important;
        border: none !important; border-radius: 4px !important; padding: 0.7rem 1.5rem !important; font-weight: 600 !important;
    }
    .stButton>button:hover { background: linear-gradient(135deg, #0B409C, #052663) !important; }
    .status-panel { background: #ffffff; border-left: 4px solid #28A745; padding: 1rem 1.5rem; border-radius: 4px; margin-bottom: 1.5rem; font-size: 0.9rem; color: #4A5568; }
    .data-card { background: #ffffff; border: 1px solid #E2E8F0; border-radius: 6px; padding: 1.25rem; text-align: center; }
    .data-value { font-size: 1.6rem; font-weight: 700; color: #1A5AC6; }
    .data-label { font-size: 0.8rem; color: #718096; margin-top: 4px; text-transform: uppercase; }
    .skeleton-placeholder { background-color: #ffffff; border: 1px solid #E2E8F0; border-radius: 8px; height: 360px; display: flex; flex-direction: column; justify-content: center; align-items: center; color: #94A3B8; font-size: 0.9rem; }
    .skeleton-line { width: 60px; height: 4px; background: #E2E8F0; margin-top: 10px; border-radius: 2px; }
    .fixed-footer { position: fixed; bottom: 0; left: 0; width: 100%; background-color: rgba(255, 255, 255, 0.92); text-align: center; padding: 14px 0; font-size: 0.8rem; color: #718096; border-top: 1px solid #E2E8F0; backdrop-filter: blur(10px); z-index: 9999; }
    .main .block-container { padding-bottom: 100px !important; }
</style>
""", unsafe_allow_html=True)

# ---------- 1. 独立紧凑图标列 (收缩时的迷你侧边栏) ----------
st.markdown(f"""
<div class="collapsed-rail">
    <a href="?task=0" target="_self" class="rail-item {'active' if st.session_state.task_index == 0 else ''}" title="风格迁移：街景转动漫"><div class="css-icon icon-anime"></div></a>
    <a href="?task=1" target="_self" class="rail-item {'active' if st.session_state.task_index == 1 else ''}" title="实体转换：苹果与橘子"><div class="css-icon icon-fruit"></div></a>
    <a href="?task=2" target="_self" class="rail-item {'active' if st.session_state.task_index == 2 else ''}" title="风格迁移：现实与莫奈"><div class="css-icon icon-monet"></div></a>
    <a href="?task=3" target="_self" class="rail-item {'active' if st.session_state.task_index == 3 else ''}" title="物理仿真：手机与单反景深"><div class="css-icon icon-dslr"></div></a>
    <a href="?task=4" target="_self" class="rail-item {'active' if st.session_state.task_index == 4 else ''}" title="自定义：加载本地 ONNX 模型"><div class="css-icon icon-custom"></div></a>
</div>
""", unsafe_allow_html=True)

# ---------- 2. 顶部标准导航大厅 ----------
st.markdown("""
<div class="platform-header">
    <div class="platform-brand">
        <div class="platform-logo">CORE-ENGINE</div>
        <div class="platform-title">多模态图像数字处理与仿真平台</div>
    </div>
    <div>
        <span style="background-color: #E6F4EA; color: #137333; padding: 6px 14px; border-radius: 4px; font-size: 0.85rem; font-weight: 600;">
            架构编译中心已就绪
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- 3. 左侧控制台布局 (展开状态呈现平铺块) ----------
with st.sidebar:
    st.markdown("### 核心任务空间")

    st.markdown(f"""
    <div class="flat-menu-container">
        <a href="?task=0" target="_self" class="menu-item {'active' if st.session_state.task_index == 0 else ''}">
            <div class="css-icon icon-anime"></div>
            <span class="menu-text">风格迁移：街景转动漫 (AnimeGAN)</span>
        </a>
        <a href="?task=1" target="_self" class="menu-item {'active' if st.session_state.task_index == 1 else ''}">
            <div class="css-icon icon-fruit"></div>
            <span class="menu-text">实体转换：苹果与橘子 (CycleGAN)</span>
        </a>
        <a href="?task=2" target="_self" class="menu-item {'active' if st.session_state.task_index == 2 else ''}">
            <div class="css-icon icon-monet"></div>
            <span class="menu-text">风格迁移：现实与莫奈 (CycleGAN)</span>
        </a>
        <a href="?task=3" target="_self" class="menu-item {'active' if st.session_state.task_index == 3 else ''}">
            <div class="css-icon icon-dslr"></div>
            <span class="menu-text">物理仿真：手机与单反景深 (CycleGAN)</span>
        </a>
        <a href="?task=4" target="_self" class="menu-item {'active' if st.session_state.task_index == 4 else ''}">
            <div class="css-icon icon-custom"></div>
            <span class="menu-text">自定义：加载本地 ONNX 模型</span>
        </a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    model_path = None
    resize_target = (256, 256)

    if st.session_state.task_index == 0:
        model_path = "onnx_models/anime_style.onnx"
    elif st.session_state.task_index == 1:
        direction = st.radio("数据流向", ["苹果 -> 橘子", "橘子 -> 苹果"])
        model_path = "onnx_models/apple_to_orange.onnx" if "苹果" in direction.split("->")[0] else "onnx_models/orange_to_apple.onnx"
    elif st.session_state.task_index == 2:
        direction = st.radio("数据流向", ["照片 -> 莫奈画作", "莫奈画作 -> 照片"])
        model_path = "onnx_models/photo_to_monet.onnx" if "照片" in direction.split("->")[0] else "onnx_models/monet_to_photo.onnx"
    elif st.session_state.task_index == 3:
        direction = st.radio("数据流向", ["手机原图 -> 单反景深", "单反景深 -> 手机原图"])
        model_path = "onnx_models/iphone_to_dslr.onnx" if "手机" in direction.split("->")[0] else "onnx_models/dslr_to_iphone.onnx"
    elif st.session_state.task_index == 4:
        uploaded_model = st.file_uploader("上传自定义 ONNX 权重文件", type=["onnx"])
        if uploaded_model:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".onnx")
            tfile.write(uploaded_model.read())
            tfile.close()  # 🔴 关键修复：关闭文件读写句柄，释放 Windows 文件锁
            model_path = tfile.name
            st.success("外部计算图载入成功")

    st.markdown("---")
    st.markdown("### 架构运行说明")
    st.caption("本平台模型均已转换为静态计算图，独立于传统重型深度学习环境。数据处理全流程在本地节点完成。")

# ---------- 4. 主页面内容排版 ----------
st.markdown("""
<div class="status-panel">
    <strong>[运行就绪]</strong> 推理服务状态正常。您可以点击左上角的收缩按钮，体验极简沉浸的工作流模式。
</div>
""", unsafe_allow_html=True)

uploaded_img = st.file_uploader("选择或拖拽本地图像文件进行转换 (支持 JPG / JPEG / PNG)", type=["jpg", "png", "jpeg"],
                                label_visibility="collapsed")

def run_onnx_engine(model_path, pil_image, resize_to):
    if "anime" in model_path.lower():
        w, h = pil_image.size
        img = pil_image.resize(((w // 32) * 32, (h // 32) * 32), Image.Resampling.LANCZOS)
    else:
        img = pil_image.resize(resize_to, Image.Resampling.LANCZOS)

    img_data = np.array(img).astype(np.float32)
    img_data = (img_data / 127.5) - 1.0
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0)

    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: img_data})[0]

    result = np.squeeze(result, axis=0)
    result = (result + 1.0) * 127.5
    result = np.clip(result, 0, 255).astype(np.uint8)
    return Image.fromarray(np.transpose(result, (1, 2, 0)))

if uploaded_img is not None:
    content_img = Image.open(uploaded_img).convert("RGB")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("<h4 style='color:#0B409C; margin-bottom:10px;'>输入端 (Source Input)</h4>", unsafe_allow_html=True)
        st.image(content_img, use_container_width=True)
        trigger_button = st.button("开始推理计算", use_container_width=True, type="primary")

    with col2:
        st.markdown("<h4 style='color:#0B409C; margin-bottom:10px;'>输出端 (Target Output)</h4>", unsafe_allow_html=True)
        if trigger_button:
            if not model_path or not os.path.exists(model_path):
                st.error("未检测到模型文件，请检查文件路径或重新上传。")
            else:
                with st.spinner("正在执行异构张量计算与生成映射..."):
                    start_time = time.time()
                    try:
                        final_img = run_onnx_engine(model_path, content_img, resize_target)
                        duration = time.time() - start_time

                        st.image(final_img, use_container_width=True)

                        st.markdown(f"""
                        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-top: 1rem;">
                            <div class="data-card">
                                <div class="data-value">{duration:.3f}s</div>
                                <div class="data-label">Latency</div>
                            </div>
                            <div class="data-card">
                                <div class="data-value">CPU Static</div>
                                <div class="data-label">Provider</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"运行时异构计算发生未知异常: {e}")
        else:
            st.markdown("""
            <div class="skeleton-placeholder">
                <span>等待流水线激活指令</span>
                <div class="skeleton-line"></div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="background-color: #ffffff; border: 1px solid #E2E8F0; border-radius: 8px; padding: 5rem 2rem; text-align: center; margin-top: 1rem;">
        <h3 style="color: #0B409C; font-weight:600;">系统数据流空置</h3>
        <p style="color: #718096; max-width: 550px; margin: 0.5rem auto 0 auto; font-size: 0.95rem; line-height:1.6;">
            请在上方控制区内上传您需要处理的图像资产。系统会自动解析空间维度，并启动多模态矩阵流进行静态计算图推理。
        </p>
    </div>
    """, unsafe_allow_html=True)

# ---------- 5. 底部固定悬浮 Footer ----------
st.markdown("""
<div class="fixed-footer">
    © 2026 多模态图像数字处理与仿真平台 · 基于 ONNX Runtime 跨平台轻量化推理规范
</div>
""", unsafe_allow_html=True)