import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
from model import PI_UNet_Transient

st.set_page_config(page_title="地下水替代模型管控平台", layout="wide")
st.title("🌊 地下水溶质运移替代模型系统")


# ================= 1. 资源全局缓存 (核心优化) =================
# 使用 st.cache_resource 确保模型和底图只加载一次，拖动滑块时不会重复加载
@st.cache_resource
def load_assets():
    # Hugging Face 免费版只有 CPU，强制使用 CPU
    device = torch.device("cpu")

    # 加载模型
    model = PI_UNet_Transient(in_channels=23, out_channels=10).to(device)
    weights_path = "weights/final_pi_unet_model.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # 加载底图
    static_data = np.load("static_data/static_base_input.npy")

    return model, static_data, device


try:
    MODEL, STATIC_BASE_INPUT, DEVICE = load_assets()
except Exception as e:
    st.error(f"资源加载失败，请检查 weights 和 static_data 文件夹是否上传完整！错误: {e}")
    st.stop()


# ================= 2. 物理场生成逻辑 =================
def generate_channel_22(sources_list, local_shape=(96, 96), sigma=3.0):
    gauss_map = np.zeros(local_shape, dtype=np.float32)
    y_indices, x_indices = np.indices(local_shape)
    xmin, xmax = 637685.10, 642753.32
    ymin, ymax = 4388618.43, 4393090.40

    for src in sources_list:
        xs_px = (src['x'] - xmin) / (xmax - xmin) * local_shape[1]
        ys_px = (ymax - src['y']) / (ymax - ymin) * local_shape[0]
        current_gauss = src['m'] * np.exp(-((x_indices - xs_px) ** 2 + (y_indices - ys_px) ** 2) / (2 * sigma ** 2))
        gauss_map += current_gauss
    return gauss_map


# ================= 3. 页面交互设计 =================
if "concentration_matrix" not in st.session_state:
    st.session_state.concentration_matrix = None

with st.sidebar:
    st.header("⚙️ 污染源输入参数配置")
    default_sources = pd.DataFrame([
        {"x": 638000.0, "y": 439000.0, "m": 150.0},
        {"x": 641000.0, "y": 438800.0, "m": 200.0}
    ])
    edited_df = st.data_editor(default_sources, num_rows="dynamic")
    start_sim = st.button("🚀 启动模型模拟", use_container_width=True)

if start_sim:
    sources_dict = edited_df.to_dict(orient="records")
    with st.spinner(">> 正在调用 PI-UNet 模型进行张量推演..."):
        # 1. 生成源项并拼接
        channel_22 = generate_channel_22(sources_dict)
        full_input_np = np.concatenate([STATIC_BASE_INPUT, channel_22[np.newaxis, ...]], axis=0)
        full_input_tensor = torch.tensor(full_input_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # 2. 模型推理
        with torch.no_grad():
            output_tensor = MODEL(full_input_tensor)

        # 3. 反变换
        output_np = output_tensor.numpy()[0]
        real_conc = np.power(10.0, output_np) - 1.0
        real_conc = np.where(real_conc < 1e-4, 0.0, real_conc)

        st.session_state.concentration_matrix = np.round(real_conc, 4)

# ================= 4. 结果展示 =================
if st.session_state.concentration_matrix is not None:
    st.success("✅ 推演成功！已获取全时空网格浓度数据。")
    st.subheader("📊 空间网格浓度数值查看")

    t_step = st.slider("选择查看的时间步 (Time Step)", min_value=1, max_value=10, value=1) - 1
    current_time_data = st.session_state.concentration_matrix[t_step]

    df_show = pd.DataFrame(
        current_time_data,
        columns=[f"Col_{i}" for i in range(96)],
        index=[f"Row_{i}" for i in range(96)]
    )
    st.dataframe(df_show, height=600)