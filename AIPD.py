import streamlit as st
from streamlit_extras.colored_header import colored_header
from streamlit_option_menu import option_menu
from streamlit_extras.badges import badge
from streamlit_shap import st_shap
from streamlit_card import card
from streamlit_drawable_canvas import st_canvas
import threading
from multiprocessing import freeze_support
import time
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.signal import savgol_filter
import colour
from colour import SpectralDistribution, XYZ_to_xy
from colour.plotting import plot_chromaticity_diagram_CIE1931
from sklearn.model_selection import train_test_split as TTS
from sklearn.model_selection import cross_val_score as CVS
from sklearn.model_selection import cross_validate as CV
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn import tree
from sklearn import svm
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LinearRegression as LinearR
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RationalQuadratic, CompoundKernel, \
Exponentiation,ConstantKernel, ExpSineSquared, Hyperparameter, Kernel, Matern, PairwiseKernel, Product, RationalQuadratic, RBF, Sum
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_regression as MIR
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from adapt.instance_based import TrAdaBoostR2
from adapt.instance_based import TwoStageTrAdaBoostR2
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
import Bgolearn.BGOsampling as BGOS
from bayes_opt import BayesianOptimization
from typing import Optional
import graphviz
import shap
import pickle
from sklearn.gaussian_process.kernels import RBF
import warnings
from prettytable import PrettyTable
import scienceplots
import base64
import uuid
import re
import io
import json
from utils import *
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score




def download_button(object_to_download, download_filename, button_text, pickle_it=False):
    if pickle_it:
        try:
            object_to_download = pickle.dumps(object_to_download)
        except pickle.PicklingError as e:
            st.write(e)
            return None
    else:
        if isinstance(object_to_download, bytes):
            pass

        elif isinstance(object_to_download, pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False)

        # Try JSON encode for everything else
        else:
            object_to_download = pickle.dumps(object_to_download)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()

    except AttributeError as e:
        b64 = base64.b64encode(object_to_download).decode()

    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)

    prim_color = '#ffffff'
    bg_color1 = '#66CDAA'
    bg_color2 = '#66CDAA'
    sbg_color = '#111111'
    txt_color = '#111111'
    font = 'Microsoft YaHei'


    custom_css = f"""
        <style>
            #{button_id} {{
                background-color: {bg_color1};
                color: {txt_color};
                padding: 0.25rem 0.75rem;
                position: relative;
                line-height: 1.6;
                border-radius: 0.25rem;
                border-width: 1px;
                border-style: solid;
                border-color: #ffffff;
                border-image: initial;
                filter: brightness(105%);
                justify-content: center;
                margin: 0px;
                width: auto;
                appearance: button;
                display: inline-flex;
                family-font: {font};
                font-weight: 400;
                letter-spacing: normal;
                word-spacing: normal;
                text-align: center;
                text-rendering: auto;
                text-transform: none;
                text-indent: 0px;
                text-shadow: none;
                text-decoration: none;
            }}
            #{button_id}:hover {{

                border-color: {prim_color};
                color: #ffff00;
                background-color: {bg_color2};
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: #ff0066;
                color: {sbg_color};
                }}
        </style> """

    dl_link = custom_css + f'<a download="{download_filename}" class= "" id="{button_id}" ' \
                           f'href="data:file/txt;base64,{b64}">{button_text}</a><br></br>'

    return dl_link



# # 设置页面标题
# st.set_page_config(page_title="AIPD", layout="wide")

# # 侧边栏
# with st.sidebar:
#     select_option = option_menu("功能菜单",
#                                 ["首页", "光谱图像数据提取", "数据库", "数据预处理", "特征工程",
#                                  "模型建立", "AI自动化建模", "SHAP-模型可解释工具"
#                                  ])
#
# if select_option == "首页":
#
#     colored_header(label="无机发光材料AI辅助设计平台（AIPD）",
#                    description="AIPD是一种无需编程的无机发光材料设计平台，采用模块化架构，集成自动化建模流程，可实现发光材料性能预测及组分优化。",
#                    color_name="blue-90")
#
#     colored_header(label="数据格式", description="数据只支持`.csv`文件", color_name="blue-90")

import streamlit as st
from streamlit_option_menu import option_menu

# 设置页面
st.set_page_config(
    page_title="无机发光材料AI辅助设计平台",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .platform-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .platform-icon {
        font-size: 4rem;
        margin-bottom: 0.5rem;
    }
    .feature-icon {
        font-size: 2rem;
        margin-right: 0.8rem;
        vertical-align: middle;
    }
    .feature-card {
        padding: 1.5rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        margin-bottom: 1.5rem;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    }
    .sidebar-icon {
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# 定义统一的图标映射
ICON_MAPPING = {
    "首页": "house",
    "光谱图像数据提取": "image",
    "数据库": "database",
    "数据预处理": "gear",
    "特征工程": "wrench",
    "模型建立": "diagram-3",
    "AI自动化建模": "robot",
    "SHAP-模型可解释工具": "graph-up"
}

# 定义图标显示名称映射（用于首页显示）
ICON_DISPLAY_MAPPING = {
    "house": "🏠",
    "image": "🖼️",
    "database": "💾",
    "gear": "⚙️",
    "wrench": "🔧",
    "diagram-3": "📊",
    "robot": "🤖",
    "graph-up": "🔍"
}

# 侧边栏菜单
with st.sidebar:
    st.markdown("""
    <div class="sidebar-icon">✨AIPD</div>
    <div style="text-align: center; margin-bottom: 2rem;">
    </div>
    """, unsafe_allow_html=True)

    select_option = option_menu(
        "功能菜单",
        ["首页", "光谱图像数据提取", "数据库", "数据预处理", "特征工程",
         "模型建立", "AI自动化建模", "SHAP-模型可解释工具"],
        styles={
            "container": {"font-family": "Arial, sans-serif", "font-size": "16px"},
            "nav-link": {"font-size": "26px"},
            "nav-link-selected": {"font-size": "30px", "font-weight": "bold"}
        },
        icons=[ICON_MAPPING[option] for option in [
            "首页", "光谱图像数据提取", "数据库", "数据预处理", "特征工程",
            "模型建立", "AI自动化建模", "SHAP-模型可解释工具"
        ]],
        menu_icon="cast",
        default_index=0
    )

# 首页内容
if select_option == "首页":
    # 平台标题
    st.markdown("""
    <div class="platform-header">
        <h1>无机发光材料AI辅助设计平台（AIPD）</h1>
        <p>AIPD是一种无需编程的无机发光材料设计平台，采用模块化架构，集成自动化建模流程，可实现发光材料性能预测及组分优化。</p>
    </div>
    """, unsafe_allow_html=True)

    # 功能模块介绍
    st.header("🚀 平台功能模块")

    # 使用两列布局展示功能模块
    col1, col2 = st.columns(2)

    with col1:
        # 首页
        st.markdown(f"""
        <div class="feature-card">
            <span class="feature-icon">{ICON_DISPLAY_MAPPING['house']}</span>
            <strong style="font-size: 28px;">首页</strong>
            <p  style="font-size: 22px;">平台概览和功能导览，提供快速开始指南和系统介绍。</p>
        </div>
        """, unsafe_allow_html=True)

        # 数据库
        st.markdown(f"""
        <div class="feature-card">
            <span class="feature-icon">{ICON_DISPLAY_MAPPING['database']}</span>
            <strong  style="font-size: 28px;">数据库</strong>
            <p style="font-size: 22px;">内置无机发光材料数据库，包含多种无机发光材料的组成、结构和性能数据。</p>
        </div>
        """, unsafe_allow_html=True)

        # 特征工程
        st.markdown(f"""
        <div class="feature-card">
            <span class="feature-icon">{ICON_DISPLAY_MAPPING['wrench']}</span>
            <strong  style="font-size: 28px;">特征工程</strong>
            <p  style="font-size: 22px;">可视化特征提取、特征变换和特征选择工具，帮助提取最重要的特征变量。</p>
        </div>
        """, unsafe_allow_html=True)

        # AI自动化建模
        st.markdown(f"""
        <div class="feature-card">
            <span class="feature-icon">{ICON_DISPLAY_MAPPING['robot']}</span>
            <strong style="font-size: 28px;">AI自动化建模</strong>
            <p style="font-size: 22px;">自动化机器学习流程，包括自动算法选择、超参数优化和模型评估。</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # 光谱图像数据提取
        st.markdown(f"""
        <div class="feature-card">
            <span class="feature-icon">{ICON_DISPLAY_MAPPING['image']}</span>
            <strong style="font-size: 28px;">光谱图像数据提取</strong>
            <p style="font-size: 22px;">从光谱图像中自动提取数值数据，支持多种图像格式，能够实现光谱图像曲线提取与色度分析。</p>
        </div>
        """, unsafe_allow_html=True)

        # 数据预处理
        st.markdown(f"""
        <div class="feature-card">
            <span class="feature-icon">{ICON_DISPLAY_MAPPING['gear']}</span>
            <strong style="font-size: 28px;">数据预处理</strong>
            <p style="font-size: 22px;">数据分布可视化、异常值检测与清洗。
            </p>
        </div>
        """, unsafe_allow_html=True)

        # 模型建立
        st.markdown(f"""
        <div class="feature-card">
            <span class="feature-icon">{ICON_DISPLAY_MAPPING['diagram-3']}</span>
            <strong style="font-size: 28px;">模型建立</strong>
            <p style="font-size: 22px;">提供多种机器学习回归算法，用于构建无机材料性能预测模型。</p>
        </div>
        """, unsafe_allow_html=True)

        # SHAP-模型可解释工具
        st.markdown(f"""
        <div class="feature-card">
            <span class="feature-icon">{ICON_DISPLAY_MAPPING['graph-up']}</span>
            <strong style="font-size: 28px;">SHAP-模型可解释工具</strong>
            <p style="font-size: 22px;">使用SHAP值解释模型预测结果，分析各特征对预测结果的贡献度。</p>
        </div>
        """, unsafe_allow_html=True)

    # 数据格式说明
    st.markdown("## 📁 数据格式要求")
    st.info("""
    - 第一行应为列名（名称）
    - 缺失值请用空单元格或NA标识
    - 建议使用UTF-8编码以避免中文乱码问题
    """)

    # 平台特色
    st.markdown("## 🌟 平台特色")
    features = st.columns(3)
    with features[0]:
        st.metric("自动化程度", "90%")
        st.caption("自动化建模流程")
    with features[1]:
        st.metric("算法数量", "15+")
        st.caption("机器学习算法")
    with features[2]:
        st.metric("材料数据", "1000+")
        st.caption("预置材料样本")


elif select_option == "光谱图像数据提取":
    with st.sidebar:
        sub_option = option_menu(None, ["坐标点提取", "光谱提取与色度分析"])


    if sub_option == "坐标点提取":


        colored_header(label="光谱图像坐标点选取工具", description=" ",
                       color_name="blue-90")

        # 初始化会话状态
        if 'points' not in st.session_state:
            st.session_state.points = []
        if 'selected_index' not in st.session_state:
            st.session_state.selected_index = 0
        if 'image_size' not in st.session_state:
            st.session_state.image_size = (0, 0)

        uploaded_file = st.file_uploader(
            "上传光谱图像",
            type=["png", "jpg", "jpeg"],
            help="支持PNG、JPG和JPEG格式"
        )

        if uploaded_file:
            # 设置参数列
            col1, col2, col3 = st.columns(3)

            with col1:
                zoom = st.slider(
                    "放大倍率",
                    min_value=1.0,
                    max_value=5.0,
                    value=2.0,
                    step=0.1,
                    help="调整放大镜的放大倍数"
                )

            with col2:
                magnifier_size = st.slider(
                    "放大镜尺寸",
                    min_value=50,
                    max_value=200,
                    value=100,
                    step=10,
                    help="调整放大镜的显示大小(像素)"
                )

            with col3:
                max_points = st.number_input(
                    "最大点数",
                    min_value=1,
                    max_value=10,
                    value=6,
                    help="设置需要选取的坐标点数量"
                )

            # 主图像处理区域
            try:
                # 图像处理
                image = Image.open(uploaded_file)
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                img_width, img_height = image.size
                st.session_state.image_size = (img_width, img_height)

                # 显示图像信息
                st.info(f"图像尺寸: {img_width} × {img_height} 像素 | 格式: {image.format}")

                # HTML/JavaScript组件
                html_code = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <style>
                        #magnifier-container {{
                            position: relative;
                            width: 100%;
                            max-width: {min(1800, img_width)}px;
                            margin: 20px auto;
                        }}
                        #main-image {{
                            width: 100%;
                            max-width: 100%;
                            height: auto;
                            cursor: crosshair;
                            border: 2px solid #e0e0e0;
                            border-radius: 8px;
                            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                        }}
                        #magnifier {{
                            position: absolute;
                            border: 2px solid #555;
                            border-radius: 50%;                    
                            width: {magnifier_size}px;
                            height: {magnifier_size}px;
                            overflow: hidden;
                            display: none;
                            z-index: 10;
                            box-shadow: 0 0 15px rgba(0,0,0,0.4);
                            pointer-events: none;
                            transform: translate(-50%, -50%);
                        }}
                        #magnifier img {{
                            position: absolute;
                            width: {img_width}px;
                            height: {img_height}px;
                            transform-origin: 0 0;
                        }}
                        .point-marker {{
                            position: absolute;
                            width: 14px;
                            height: 14px;
                            border-radius: 50%;
                            transform: translate(-7px, -7px);
                            z-index: 5;
                            border: 2px solid white;
                            box-shadow: 0 0 5px rgba(0,0,0,0.7);
                        }}
                        .point-label {{
                            position: absolute;
                            background: rgba(255,255,255,0.95);
                            padding: 4px 10px;
                            border-radius: 15px;
                            font-size: 12px;
                            font-weight: bold;
                            transform: translate(10px, -10px);
                            z-index: 5;
                            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
                            white-space: nowrap;
                        }}
                        #status-info {{
                            margin: 15px 0;
                            padding: 12px;
                            background: #f0f8ff;
                            border-radius: 6px;
                            font-weight: bold;
                            color: #2c3e50;
                            border-left: 4px solid #2196F3;
                            text-align: center;
                        }}
                    </style>
                </head>
                <body>
                    <div id="magnifier-container">
                        <img id="main-image" src="data:image/png;base64,{img_base64}"/>
                        <div id="magnifier"><img src="data:image/png;base64,{img_base64}" /></div>
                    </div>
                    <div id="status-info">
                        {f"🖱️ 点击图像添加坐标点 (最多{max_points}个点)" if not st.session_state.points
                else f"✅ 已选择 {len(st.session_state.points)}/{max_points} 个点"}
                    </div>
    
                    <script>
                        // 初始化变量
                        const magnifier = document.getElementById("magnifier");
                        const mainImage = document.getElementById("main-image");
                        const zoomLevel = {zoom};
                        const maxPoints = {max_points};
                        let points = {json.dumps([[p[0], p[1]] for p in st.session_state.points])};
                        const imgWidth = {img_width};
                        const imgHeight = {img_height};
    
                        // 更新点标记
                        function updatePointMarkers() {{
                            // 清除旧标记
                            document.querySelectorAll('.point-marker, .point-label').forEach(el => el.remove());
    
                            // 添加新标记
                            points.forEach(function(point, index) {{
                                const x = point[0];
                                const y = point[1];
    
                                // 创建点标记
                                const marker = document.createElement('div');
                                marker.className = 'point-marker';
                                marker.style.left = x + 'px';
                                marker.style.top = y + 'px';
                                marker.style.backgroundColor = index === 0 ? '#4CAF50' : 
                                                              index === 1 ? '#F44336' : 
                                                              '#2196F3';
                                document.getElementById('magnifier-container').appendChild(marker);
    
                                // 创建点标签
                                const label = document.createElement('div');
                                label.className = 'point-label';
                                label.style.left = x + 'px';
                                label.style.top = y + 'px';
                                label.textContent = '点' + (index + 1) + ' (' + x + ', ' + y + ')';
                                document.getElementById('magnifier-container').appendChild(label);
                            }});
    
                            // 更新状态信息
                            const statusInfo = document.getElementById("status-info");
                            if (points.length === 0) {{
                                statusInfo.innerHTML = '🖱️ 点击图像添加坐标点 (最多' + maxPoints + '个点)';
                                statusInfo.style.background = '#fff3cd';
                                statusInfo.style.borderLeftColor = '#ffc107';
                            }} else if (points.length < maxPoints) {{
                                statusInfo.innerHTML = '✅ 已选择 ' + points.length + '/' + maxPoints + ' 个点 - 继续添加';
                                statusInfo.style.background = '#d4edda';
                                statusInfo.style.borderLeftColor = '#28a745';
                            }} else {{
                                statusInfo.innerHTML = '已完成 ' + points.length + '/' + maxPoints + ' 个点的选择';
                                statusInfo.style.background = '#d1ecf1';
                                statusInfo.style.borderLeftColor = '#17a2b8';
                            }}
    
                            // 发送数据回Streamlit
                            window.parent.postMessage({{
                                type: 'streamlit:setComponentValue',
                                value: points,
                                imageWidth: imgWidth,
                                imageHeight: imgHeight
                            }}, '*');
                        }}
    
                        // 初始化显示
                        updatePointMarkers();
    
                        // 鼠标移动放大镜效果
                        mainImage.addEventListener("mousemove", function(e) {{
                            const rect = mainImage.getBoundingClientRect();
                            const scaleX = mainImage.offsetWidth / imgWidth;
                            const scaleY = mainImage.offsetHeight / imgHeight;
    
                            // 计算实际图像坐标
                            const x = (e.clientX - rect.left) / scaleX;
                            const y = (e.clientY - rect.top) / scaleY;
    
                            // 边界检查
                            if (x < 0 || y < 0 || x > imgWidth || y > imgHeight) {{
                                magnifier.style.display = "none";
                                return;
                            }}
    
                            // 显示放大镜并精确定位中心
                            magnifier.style.display = "block";
                            magnifier.style.left = (e.clientX - rect.left) + 'px';
                            magnifier.style.top = (e.clientY - rect.top) + 'px';
    
                            // 计算放大图像位置
                            const zoomedImg = magnifier.querySelector("img");
                            const magnifierSize = magnifier.offsetWidth;
    
                            // 计算放大后的偏移量，确保鼠标位置在放大镜中心
                            const offsetX = (x * zoomLevel) - (magnifierSize / 2 / scaleX);
                            const offsetY = (y * zoomLevel) - (magnifierSize / 2 / scaleY);
    
                            zoomedImg.style.transform = `scale(${{zoomLevel}})`;
                            zoomedImg.style.left = `-${{offsetX}}px`;
                            zoomedImg.style.top = `-${{offsetY}}px`;
                        }});
    
                        // 鼠标离开隐藏放大镜
                        mainImage.addEventListener("mouseleave", function() {{
                            magnifier.style.display = "none";
                        }});
    
                        // 点击事件处理
                        mainImage.addEventListener("click", function(e) {{
                            const rect = mainImage.getBoundingClientRect();
                            const scaleX = mainImage.offsetWidth / imgWidth;
                            const scaleY = mainImage.offsetHeight / imgHeight;
    
                            // 计算实际图像坐标
                            const x = Math.round((e.clientX - rect.left) / scaleX);
                            const y = Math.round((e.clientY - rect.top) / scaleY);
    
                            // 检查是否已达最大点数
                            if (points.length >= maxPoints) {{
                                alert('已达到最大点数 ' + maxPoints + '！');
                                return;
                            }}
    
                            // 添加新点
                            points.push([x, y]);
                            updatePointMarkers();
                        }});
                    </script>
                </body>
                </html>
                """

                # 渲染HTML组件
                st.components.v1.html(html_code, height=min(1600, img_height + 150))

                # 显示当前选择的点
                if st.session_state.points:
                    st.subheader("已选坐标点")
                    points_df = pd.DataFrame(st.session_state.points, columns=['X坐标', 'Y坐标'])
                    points_df.index = [f'点{i + 1}' for i in range(len(st.session_state.points))]
                    st.dataframe(points_df, use_container_width=True)

            except Exception as e:
                st.error(f"图像处理错误: {str(e)}")


    elif sub_option == "光谱提取与色度分析":

        # 中文显示设置
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

        warnings.filterwarnings("ignore", category=UserWarning)

        colored_header(label="光谱提取与色度分析", description=" ",
                       color_name="blue-90")


        # 添加化学组成式和晶体结构输入框
        # st.header("材料信息")
        colored_header(label="材料信息", description="", color_name="blue-70")

        col_chem, col_crystal = st.columns(2)
        with col_chem:
            chemical_formula = st.text_input("化学组成式", placeholder="例如: YAG:Ce")
        with col_crystal:
            crystal_structure = st.text_input("晶体结构", placeholder="例如: 立方晶系")

        # 第一部分：光谱曲线提取
        st.header("第一步：光谱曲线提取")
        uploaded_file = st.file_uploader("上传光谱图像", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            # 图像处理函数
            def process_image(uploaded_file):
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


            image_rgb = process_image(uploaded_file)
            img_h, img_w = image_rgb.shape[:2]

            st.subheader("① 手动输入坐标区域")
            st.markdown("""
            <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">
                🟢 请输入坐标范围值<br>
                - X_min: 最小波长对应的x坐标<br>
                - X_max: 最大波长对应的x坐标<br>
                - Y_min: 最小强度对应的y坐标<br>
                - Y_max: 最大强度对应的y坐标
            </div>
            """, unsafe_allow_html=True)

            # 显示图像供参考
            st.image(image_rgb, caption="参考图像", use_column_width=True)

            # 创建四个输入框
            col1, col2 = st.columns(2)
            with col1:
                x_min = st.number_input("X_min (最小x坐标)", min_value=0, max_value=img_w, value=0)
                x_max = st.number_input("X_max (最大x坐标)", min_value=0, max_value=img_w, value=img_w)
            with col2:
                y_min = st.number_input("Y_min (最小y坐标)", min_value=0, max_value=img_h, value=0)
                y_max = st.number_input("Y_max (最大y坐标)", min_value=0, max_value=img_h, value=img_h)

            # 验证输入的有效性
            if x_min >= x_max:
                st.error("X_min必须小于X_max")
            elif y_min >= y_max:
                st.error("Y_min必须小于Y_max")
            else:
                st.success(f"✅ 坐标区域：X = [{x_min}, {x_max}], Y = [{y_min}, {y_max}]")

                st.subheader("② 设置坐标轴物理范围")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    wl_min = st.number_input("波长最小值 (nm)", value=575.0, min_value=300.0, max_value=800.0)
                with col2:
                    wl_max = st.number_input("波长最大值 (nm)", value=700.0, min_value=wl_min + 1, max_value=1000.0)
                with col3:
                    inten_min = st.number_input("强度最小值", value=0.0)
                with col4:
                    inten_max = st.number_input("强度最大值", value=1.0, min_value=inten_min + 0.1)

                st.subheader("③ 用画笔涂选想要提取的区域（大概圈出即可）")
                draw_result = st_canvas(
                    fill_color="rgba(255, 0, 0, 0.3)",
                    stroke_width=st.slider("🖌️ 画笔粗细", 1, 50, 5),
                    stroke_color="#ff0000",
                    background_image=Image.fromarray(image_rgb),
                    update_streamlit=True,
                    height=img_h,
                    width=img_w,
                    drawing_mode="freedraw",
                    key="draw_canvas"
                )

                st.subheader("④ 设置提取参数")
                col_color, col_tol = st.columns(2)
                with col_color:
                    color_to_extract = st.color_picker("🎯 目标曲线颜色", "#000000")
                with col_tol:
                    tolerance = st.slider("🎚️ 提取颜色容差", 0, 100, 30,
                                          help="值越大，提取越密集")

                # 插值方法选择
                interp_method = st.selectbox("📈 拟合/插值方法", [
                    "无",
                    "线性插值",
                    "样条插值（spline）",
                    "三次插值（cubic）",
                    "多项式拟合（Polynomial）",
                    "Savitzky–Golay 平滑（Savgol）"
                ])

                interp_params = {}
                if interp_method == "样条插值（spline）":
                    interp_params['s'] = st.slider("样条平滑因子 s", 0.0, 5.0, 0.5, 0.1,
                                                   help="值越大，曲线越平滑")
                elif interp_method == "多项式拟合（Polynomial）":
                    interp_params['deg'] = st.slider("多项式阶数", 1, 20, 5,
                                                     help="阶数越高，拟合越精确但可能过拟合")
                elif interp_method == "Savitzky–Golay 平滑（Savgol）":
                    interp_params['window_length'] = st.slider("窗口长度（奇数）", 5, 199, 15, step=2,
                                                               help="窗口越大，平滑效果越强")
                    interp_params['polyorder'] = st.slider("多项式阶数", 1, 6, 3,
                                                           help="窗口内拟合的多项式阶数")

                # 设置输出波长范围选项
                st.subheader("⑤ 设置输出参数")
                col5, col6, col7 = st.columns(3)
                with col5:
                    output_wl_min = st.number_input("输出波长最小值 (nm)", value=wl_min,
                                                    min_value=wl_min, max_value=wl_max)
                with col6:
                    output_wl_max = st.number_input("输出波长最大值 (nm)", value=wl_max,
                                                    min_value=output_wl_min, max_value=wl_max)
                with col7:
                    output_step = st.selectbox("输出波长步长", [0.1, 0.5, 1.0, 5], index=2)

                # 光谱峰值矫正选项
                with st.expander("⚙️ 高级选项", expanded=False):
                    do_peak_correction = st.checkbox("启用峰值矫正", value=False)
                    if do_peak_correction:
                        target_peak = st.number_input(
                            "目标峰值波长 (nm)",
                            min_value=float(output_wl_min),
                            max_value=float(output_wl_max),
                            value=630.0,
                            step=1.0,
                            help="将光谱峰值移动到此波长位置"
                        )
                        st.info("峰值矫正会将整个光谱平移，使当前峰值移动到指定位置")

                if st.button("🚀 开始提取曲线数据", type="primary"):
                    if draw_result.image_data is not None:
                        with st.spinner("正在处理图像..."):
                            # 创建红色画笔区域掩码
                            drawn_mask = draw_result.image_data.astype(np.uint8)
                            red_mask = cv2.inRange(drawn_mask[:, :, :3], (200, 0, 0), (255, 80, 80))

                            # 创建目标颜色区域掩码
                            hex_color = color_to_extract.lstrip("#")
                            target_rgb = np.array([int(hex_color[i:i + 2], 16) for i in (0, 2, 4)])
                            lower = np.clip(target_rgb - tolerance, 0, 255)
                            upper = np.clip(target_rgb + tolerance, 0, 255)
                            img_mask = cv2.inRange(image_rgb, lower, upper)

                            # 合并两个掩码
                            combined_mask = cv2.bitwise_and(img_mask, red_mask)
                            ys, xs = np.where(combined_mask == 255)

                            if len(xs) == 0:
                                st.error("⚠️ 未找到符合条件的像素点，请调整颜色或容差。")
                                st.stop()

                            # 筛选在坐标区域内的点
                            idx = (xs >= x_min) & (xs <= x_max) & (ys >= y_min) & (ys <= y_max)
                            xs, ys = xs[idx], ys[idx]

                            if len(xs) == 0:
                                st.error("⚠️ 所选区域内没有符合要求的像素点，请重新选择区域。")
                                st.stop()

                            # 计算波长和强度
                            wavelengths = wl_min + (xs - x_min) / (x_max - x_min) * (wl_max - wl_min)
                            intensities = inten_max - (ys - y_min) / (y_max - y_min) * (inten_max - inten_min)

                            # 按波长排序
                            x_sorted, y_sorted = zip(*sorted(zip(wavelengths, intensities)))

                            # 绘制原始提取结果
                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.plot(x_sorted, y_sorted, '.', markersize=1, label="原始提取")
                            ax.set_xlabel("波长 (nm)")
                            ax.set_ylabel("强度")
                            ax.legend()
                            ax.set_title("提取结果")
                            ax.grid(True)
                            st.pyplot(fig)

                            # 创建文件名前缀
                            filename_prefix = ""
                            if chemical_formula:
                                filename_prefix += f"{chemical_formula.replace(':', '_')}_"
                            if crystal_structure:
                                filename_prefix += f"{crystal_structure.replace(' ', '_')}_"
                            if not filename_prefix:
                                filename_prefix = "extracted_"

                            # 原始数据下载
                            df = pd.DataFrame({"Wavelength (nm)": x_sorted, "Intensity": y_sorted})
                            csv = df.to_csv(index=False).encode("utf-8")
                            st.download_button("📥 下载原始提取数据 CSV", data=csv,
                                               file_name=f"{filename_prefix}raw.csv",
                                               mime="text/csv")

                            # 插值处理
                            if interp_method != "无":
                                with st.spinner("正在进行插值处理..."):
                                    try:
                                        # 生成插值数据
                                        x_dense = np.linspace(min(x_sorted), max(x_sorted), 1000)

                                        if interp_method == "线性插值":
                                            f = interp1d(x_sorted, y_sorted, kind="linear")
                                            y_dense = f(x_dense)
                                        elif interp_method == "样条插值（spline）":
                                            f = UnivariateSpline(x_sorted, y_sorted, s=interp_params['s'])
                                            y_dense = f(x_dense)
                                        elif interp_method == "三次插值（cubic）":
                                            f = interp1d(x_sorted, y_sorted, kind="cubic")
                                            y_dense = f(x_dense)
                                        elif interp_method == "多项式拟合（Polynomial）":
                                            coeffs = np.polyfit(x_sorted, y_sorted, deg=interp_params['deg'])
                                            f = np.poly1d(coeffs)
                                            y_dense = f(x_dense)
                                        elif interp_method == "Savitzky–Golay 平滑（Savgol）":
                                            y_dense = savgol_filter(y_sorted,
                                                                    window_length=interp_params['window_length'],
                                                                    polyorder=interp_params['polyorder'])
                                            x_dense = np.array(x_sorted)

                                        # 绘制插值结果
                                        fig2, ax2 = plt.subplots(figsize=(10, 4))
                                        ax2.plot(x_sorted, y_sorted, '.', markersize=1, label="原始提取")
                                        ax2.plot(x_dense, y_dense, '-', color='red',
                                                 label=f"{interp_method}拟合", linewidth=2)
                                        ax2.set_xlabel("波长 (nm)")
                                        ax2.set_ylabel("强度")
                                        ax2.legend()
                                        ax2.set_title("拟合结果")
                                        ax2.grid(True)
                                        st.pyplot(fig2)

                                        # 生成按指定步长和范围的插值数据
                                        x_output = np.arange(output_wl_min, output_wl_max + output_step, output_step)

                                        if interp_method == "Savitzky–Golay 平滑（Savgol）":
                                            # 对于Savgol需要先插值到均匀网格
                                            x_temp = np.linspace(min(x_sorted), max(x_sorted), len(x_sorted))
                                            f = interp1d(x_sorted, y_sorted, kind='linear')
                                            y_temp = f(x_temp)
                                            y_smoothed = savgol_filter(y_temp,
                                                                       window_length=interp_params['window_length'],
                                                                       polyorder=interp_params['polyorder'])
                                            # 再次插值到指定网格
                                            f_output = interp1d(x_temp, y_smoothed, kind='linear',
                                                                fill_value='extrapolate')
                                        else:
                                            # 其他方法直接使用现有的插值函数
                                            f_output = interp1d(x_dense, y_dense, kind='linear',
                                                                fill_value='extrapolate')

                                        y_output = f_output(x_output)

                                        # 创建输出DataFrame
                                        df_output = pd.DataFrame({
                                            "Wavelength (nm)": x_output,
                                            "Intensity": y_output
                                        })

                                        # 峰值矫正
                                        if do_peak_correction:
                                            try:
                                                # 找到当前峰值位置
                                                current_peak_idx = df_output['Intensity'].idxmax()
                                                current_peak = df_output.loc[current_peak_idx, 'Wavelength (nm)']

                                                # 计算需要移动的波长差
                                                wavelength_shift = target_peak - current_peak

                                                # 应用平移校正
                                                df_output['Wavelength (nm)'] = df_output[
                                                                                   'Wavelength (nm)'] + wavelength_shift

                                                # 更新输出范围
                                                output_wl_min += wavelength_shift
                                                output_wl_max += wavelength_shift

                                                st.success(
                                                    f"✅ 光谱峰值已从 {current_peak:.1f}nm 移动到 {target_peak:.1f}nm (平移量: {wavelength_shift:.1f}nm)")

                                                # 显示矫正前后的对比图
                                                fig_corr, ax_corr = plt.subplots(figsize=(10, 4))
                                                ax_corr.plot(df_output['Wavelength (nm)'] - wavelength_shift,
                                                             df_output['Intensity'],
                                                             label='原始光谱')
                                                ax_corr.plot(df_output['Wavelength (nm)'],
                                                             df_output['Intensity'],
                                                             '--',
                                                             label=f'矫正后 (峰值@{target_peak}nm)')
                                                ax_corr.set_xlabel('波长 (nm)')
                                                ax_corr.set_ylabel('强度')
                                                ax_corr.set_title('光谱峰值矫正对比')
                                                ax_corr.legend()
                                                ax_corr.grid(True)
                                                st.pyplot(fig_corr)

                                            except Exception as e:
                                                st.error(f"峰值矫正出错: {str(e)}")

                                        # 下载按指定范围和步长的数据
                                        csv_output = df_output.to_csv(index=False).encode("utf-8")
                                        st.download_button(
                                            f"📥 下载插值数据 ({output_wl_min}-{output_wl_max}nm, 步长{output_step}nm)",
                                            data=csv_output,
                                            file_name=f"{filename_prefix}fitted_{output_wl_min}_{output_wl_max}_{output_step}nm.csv",
                                            mime="text/csv"
                                        )

                                        # 第二部分：色度分析
                                        st.header("第二步：色度分析")
                                        with st.spinner("正在进行色度分析..."):
                                            # 补全光谱数据到360-830nm范围
                                            full_wavelengths = np.arange(360, 831)
                                            full_df = pd.DataFrame({'Wavelength': full_wavelengths})

                                            # 合并数据，缺失值补0
                                            df_merged = pd.merge(full_df, df_output,
                                                                 left_on='Wavelength',
                                                                 right_on='Wavelength (nm)',
                                                                 how='left')
                                            df_merged['Intensity'] = df_merged['Intensity'].fillna(0)

                                            # 计算色度坐标
                                            try:
                                                spectrum = SpectralDistribution(
                                                    dict(zip(df_merged['Wavelength'], df_merged['Intensity'])),
                                                    name='Extracted Spectrum'
                                                )
                                                cmfs = colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER[
                                                    'CIE 1931 2 Degree Standard Observer']
                                                XYZ = colour.sd_to_XYZ(spectrum, cmfs)
                                                x, y = XYZ_to_xy(XYZ)

                                                st.success(f"✅ CIE 1931 色度坐标: x={x:.4f}, y={y:.4f}")

                                                # 绘制色度图
                                                fig3, ax3 = plt.subplots(figsize=(8, 6))
                                                plot_chromaticity_diagram_CIE1931(axes=ax3, standalone=False)
                                                ax3.plot(x, y, 'c^', markersize=10,
                                                         label=f'提取点 ({x:.3f}, {y:.3f})')
                                                ax3.legend()
                                                ax3.set_title('CIE 1931 色度图')
                                                st.pyplot(fig3)

                                                # 显示光谱功率分布
                                                fig4, ax4 = plt.subplots(figsize=(10, 4))
                                                ax4.plot(df_output['Wavelength (nm)'],
                                                         df_output['Intensity'],
                                                         label='提取的光谱')
                                                ax4.set_xlabel('波长 (nm)')
                                                ax4.set_ylabel('强度')
                                                ax4.set_title('光谱功率分布')
                                                ax4.legend()
                                                ax4.grid(True)
                                                st.pyplot(fig4)

                                                # 计算光谱特征参数
                                                try:
                                                    # 找到峰值波长和最大强度（整个光谱的最大值）
                                                    max_idx = np.argmax(df_output['Intensity'])
                                                    peak_wavelength = df_output['Wavelength (nm)'].iloc[max_idx]
                                                    peak_intensity = df_output['Intensity'].iloc[max_idx]

                                                    # 计算半强度值
                                                    half_max = peak_intensity / 2

                                                    # 找到所有强度大于半强度值的数据点
                                                    above_half_max = df_output['Intensity'] >= half_max

                                                    # 获取满足条件的连续区域
                                                    regions = np.where(np.diff(above_half_max.astype(int)))[0] + 1
                                                    if above_half_max.iloc[0]:
                                                        regions = np.insert(regions, 0, 0)
                                                    if above_half_max.iloc[-1]:
                                                        regions = np.append(regions, len(df_output))

                                                    regions = regions.reshape(-1, 2)

                                                    # 找出整个光谱中最左和最右的边界点
                                                    left_boundary = df_output['Wavelength (nm)'].iloc[regions[0][0]]
                                                    right_boundary = df_output['Wavelength (nm)'].iloc[
                                                        regions[-1][1] - 1]

                                                    # 在左边界区域插值求精确左边界
                                                    left_region_start = max(0, regions[0][0] - 1)
                                                    left_region_end = min(len(df_output), regions[0][1] + 1)
                                                    left_wl = np.interp(
                                                        half_max,
                                                        df_output['Intensity'][left_region_start:left_region_end][::-1],
                                                        df_output['Wavelength (nm)'][left_region_start:left_region_end][
                                                        ::-1]
                                                    )

                                                    # 在右边界区域插值求精确右边界
                                                    right_region_start = max(0, regions[-1][0] - 1)
                                                    right_region_end = min(len(df_output), regions[-1][1] + 1)
                                                    right_wl = np.interp(
                                                        half_max,
                                                        df_output['Intensity'][right_region_start:right_region_end],
                                                        df_output['Wavelength (nm)'][
                                                        right_region_start:right_region_end]
                                                    )

                                                    # 计算整个光谱的半峰宽
                                                    fwhm = right_wl - left_wl


                                                    with st.expander("📊 光谱特征参数", expanded=True):
                                                        st.markdown(f"""
                                                        **峰值波长**: {peak_wavelength:.1f} nm  
                                                        **CIE色度坐标**: x={x:.4f}, y={y:.4f}  
                                                        **半峰宽(FWHM)**: {fwhm:.1f} nm  
                                                        **左半峰点**: {left_wl:.1f} nm  
                                                        **右半峰点**: {right_wl:.1f} nm  
                                                        """)

                                                        # 绘制带标记的示意图
                                                        fig5, ax5 = plt.subplots(figsize=(10, 4))
                                                        ax5.plot(df_output['Wavelength (nm)'],
                                                                 df_output['Intensity'],
                                                                 label='光谱')
                                                        ax5.axhline(y=half_max, color='gray',
                                                                    linestyle='--', label='半峰高')
                                                        ax5.axvline(x=left_wl, color='green',
                                                                    linestyle=':', label='左半峰点')
                                                        ax5.axvline(x=right_wl, color='blue',
                                                                    linestyle=':', label='右半峰点')
                                                        ax5.plot(peak_wavelength, peak_intensity,
                                                                 'ro', label='峰值')
                                                        ax5.set_xlabel('波长 (nm)')
                                                        ax5.set_ylabel('强度')
                                                        ax5.set_title('光谱特征参数示意图')
                                                        ax5.legend()
                                                        ax5.grid(True)
                                                        st.pyplot(fig5)

                                                except Exception as e:
                                                    st.error(f"计算光谱特征参数时出错: {str(e)}")

                                            except Exception as e:
                                                st.error(f"色度计算错误: {str(e)}")

                                    except Exception as e:
                                        st.error(f"插值处理出错：{e}")
                    else:
                        st.warning("⚠️ 请先用画笔涂选想提取的区域")

        # else:
        #     st.info("ℹ️ 请上传光谱图像以开始分析")




elif select_option == "数据库":
    colored_header(label="数据分类", description=" ", color_name="blue-90")
    # 创建示例数据
    example_data = pd.DataFrame({
    '材料编号': ['P001', 'P002', 'P003', 'P004', 'P005'],
    '发光材料': ['Ca2Y0.992Zr2Al3O12Bi0.008', 'Ca8NaGd0.98P6O24F2Ce0.02', 'Ca7.99NaGd0.98P6O24F2Ce0.02Mn0.01', 'Ca7.96NaGd0.98P6O24F2Ce0.02Mn0.04', 'Ca7.92NaGd0.98P6O24F2Ce0.02Mn0.07'],
    '第一掺杂离子': ['Bi³⁺', 'Ce³⁺', 'Ce³⁺', 'Ce³⁺', 'Ce³⁺'],
    '第一掺杂浓度': [0.008, 0.02, 0.02, 0.02, 0.02],
    '第二掺杂离子': ['无', '无', 'Mn²⁺', 'Mn²⁺', 'Mn²⁺'],
    '第二掺杂浓度': [0, 0, 0.01, 0.04, 0.07],
    '测试温度_K': [298, 298, 298, 298, 298],
    '发射波长_nm': [313, 337, 337, 337, 337],
    'CIE_x坐标': [0.1948, None, 0.38, 0.45, 0.48],
    'CIE_y坐标': [0.15, None, 0.44, 0.45, 0.49],
    '激发波长_nm': [278, 303, 303, 303, 303],
    '荧光寿命_ns': [272, 25.22, 23.27, 22.07, 20.97],
    '监测波长_nm': [313, 337, 337, 337, 337]
    })

    # 显示示例数据预览
    with st.expander("📋 查看数据库数据格式"):
        st.write("示例数据预览：")
        st.dataframe(example_data, use_container_width=True)



    col1, col2, col3 = st.columns(3)
    with col1:
        df = pd.read_csv('./data/1.csv')
        st.write("紫外荧光粉")
        image = Image.open('./data/卤磷酸盐蓝粉.png')
        st.image(image, width=100, caption='')
        tmp_download_link = download_button(df, f'紫外荧光粉.csv', button_text='下载')
        st.markdown(tmp_download_link, unsafe_allow_html=True)
    with col2:
        df = pd.read_csv('./data/2.csv')
        st.write("蓝色荧光粉")
        image = Image.open('./data/卤磷酸盐蓝粉.png')
        st.image(image, width=100, caption='')
        tmp_download_link = download_button(df, f'蓝色荧光粉.csv', button_text='下载')
        st.markdown(tmp_download_link, unsafe_allow_html=True)
    with col3:
        df = pd.read_csv('./data/3.csv')
        st.write("绿色荧光粉")
        image = Image.open('./data/铝酸盐黄绿粉.png')
        st.image(image, width=100, caption='')
        tmp_download_link = download_button(df, f'绿色荧光粉.csv', button_text='下载')
        st.markdown(tmp_download_link, unsafe_allow_html=True)


    col1, col2, col3 = st.columns(3)
    with col1:
        df = pd.read_csv('./data/4.csv')
        st.write("黄橙荧光粉")
        image = Image.open('./data/氟化物红粉.png')
        st.image(image, width=100, caption='')
        tmp_download_link = download_button(df, f'黄橙荧光粉.csv', button_text='下载')
        st.markdown(tmp_download_link, unsafe_allow_html=True)

    with col2:
        df = pd.read_csv('./data/5.csv')
        st.write("红色荧光粉")
        image = Image.open('./data/氮化物红粉.png')
        st.image(image, width=100, caption='')
        tmp_download_link = download_button(df, f'红色荧光粉.csv', button_text='下载')
        st.markdown(tmp_download_link, unsafe_allow_html=True)
    with col3:
        df = pd.read_csv('./data/6.csv')
        st.write("近红外荧光粉")
        image = Image.open('./data/近红外荧光粉.png')
        st.image(image, width=100, caption='')
        tmp_download_link = download_button(df, f'近红外荧光粉.csv', button_text='下载')
        st.markdown(tmp_download_link, unsafe_allow_html=True)

elif select_option == "数据预处理":
    with st.sidebar:
        sub_option = option_menu(None, ["数据可视化", "异常值检测"])
    if sub_option == "数据可视化":

        colored_header(label="数据可视化", description=" ", color_name="blue-90")
        file = st.file_uploader("上传`.csv`文件", type=['csv'],
                                help="只支持csv格式")

        # 创建示例数据
        example_data = pd.DataFrame({
            '材料编号': ['M001', 'M002', 'M003', 'M004'],
            '特征A': [0.5, 0.6, 0.7, 0.8],
            '特征B': [0.3, 0.25, 0.2, 0.15],
            '特征C': [0.2, 0.15, 0.1, 0.05],
            '发光性能A': [100, 150, 200, 180],
            '发光性能B': [450, 460, 455, 465]
        })

        # 显示示例数据预览
        with st.expander("📋 查看示例数据格式"):
            st.write("示例数据格式预览：")
            st.dataframe(example_data, use_container_width=True)
            st.caption("请确保您的数据文件包含类似的列结构和数据格式")

        if file is not None:
            df = pd.read_csv(file)
            check_string_NaN(df)

            colored_header(label="数据信息", description=" ", color_name="blue-70")

            nrow = st.slider("rows", 1, len(df), 5)
            df_nrow = df.head(nrow)
            st.write(df_nrow)

            colored_header(label="数据统计", description=" ", color_name="blue-30")

            st.write(df.describe())

            tmp_download_link = download_button(df.describe(), f'数据统计.csv', button_text='下载')

            st.markdown(tmp_download_link, unsafe_allow_html=True)

            colored_header(label="特征和目标", description=" ", color_name="blue-70")

            target_num = st.number_input('目标数量', min_value=1, max_value=10, value=1)
            col_feature, col_target = st.columns(2)
            # features
            features = df.iloc[:, :-target_num]
            # targets
            targets = df.iloc[:, -target_num:]
            with col_feature:
                st.write(features.head())
            with col_target:
                st.write(targets.head())

            colored_header(label="特征分布", description=" ", color_name="blue-30")
            feature_selected_name = st.selectbox('**单个特征**', list(features), 1)
            feature_selected_value = features[feature_selected_name]
            plot = customPlot()
            col1, col2 = st.columns([1, 3])

            with col1:
                with st.expander("**绘图参数**"):
                    options_selected = [plot.set_title_fontsize(1), plot.set_label_fontsize(2),
                                        plot.set_tick_fontsize(3), plot.set_legend_fontsize(4),
                                        plot.set_color('曲线颜色', 6, 5), plot.set_color('柱状图颜色', 0, 6)]
            with col2:
                plot.feature_hist_kde(options_selected, feature_selected_name, feature_selected_value)

            # =========== Targets visulization ==================

            colored_header(label="目标分布", description=" ", color_name="blue-30")

            target_selected_name = st.selectbox('target', list(targets))

            target_selected_value = targets[target_selected_name]
            plot = customPlot()
            col1, col2 = st.columns([1, 3])
            with col1:
                with st.expander("**绘图参数**"):
                    options_selected = [plot.set_title_fontsize(7), plot.set_label_fontsize(8),
                                        plot.set_tick_fontsize(9), plot.set_legend_fontsize(10),
                                        plot.set_color('line color', 6, 11), plot.set_color('bin color', 0, 12)]
            with col2:
                plot.feature_hist_kde(options_selected, target_selected_name, target_selected_value)



    elif sub_option == "异常值检测":
        colored_header(label="异常值检测", description=" ", color_name="blue-90")
        file = st.file_uploader("上传`.csv`文件", type=['csv'], help="只支持csv格式")

        # 创建示例数据
        example_data = pd.DataFrame({
            '材料编号': ['M001', 'M002', 'M003', 'M004'],
            '特征A': [0.5, 0.6, 0.7, 0.8],
            '特征B': [0.3, 0.25, 0.2, 0.15],
            '特征C': [0.2, 0.15, 0.1, 0.05],
            '发光性能A': [100, 150, 200, 180],
            '发光性能B': [450, 460, 455, 465]
        })

        # 显示示例数据预览
        with st.expander("📋 查看示例数据格式"):
            st.write("示例数据格式预览：")
            st.dataframe(example_data, use_container_width=True)
            st.caption("请确保您的数据文件包含类似的列结构和数据格式")

        if file is not None:
            df = pd.read_csv(file)
            check_string_NaN(df)
            colored_header(label="数据信息", description=" ", color_name="blue-70")
            nrow = st.slider("rows", 1, len(df), 5)
            df_nrow = df.head(nrow)
            st.write(df_nrow)

            colored_header(label="特征和目标", description=" ", color_name="blue-70")

            target_num = st.number_input('目标数目', min_value=1, max_value=10, value=1)

            col_feature, col_target = st.columns(2)

            features = df.iloc[:, :-target_num]

            targets = df.iloc[:, -target_num:]
            with col_feature:
                st.write(features.head())
            with col_target:
                st.write(targets.head())

            colored_header(label="异常值检测", description=" ", color_name="blue-30")

            model_path = './models/outlier detection'

            template_alg = model_platform(model_path)

            inputs, col2 = template_alg.show()

            if inputs['model'] == 'One Class SVM':
                detector = svm.OneClassSVM(nu=inputs['nu'], kernel='rbf', gamma=inputs['gamma'])
                detector.fit(features)
                outlier = detector.predict(features)
                normal = df[outlier == 1]
                abnormal = df[outlier == -1]
                st.write('**正常样本**')
                st.write(normal)
                tmp_download_link = download_button(normal, f'正常样本.csv', button_text='下载')
                st.markdown(tmp_download_link, unsafe_allow_html=True)

                st.write('**异常样本**')
                st.write(abnormal)
                tmp_download_link = download_button(abnormal, f'异常样本.csv', button_text='下载')
                st.markdown(tmp_download_link, unsafe_allow_html=True)

            elif inputs['model'] == 'IsolationForest':
                detector = IsolationForest(n_estimators=inputs['n_estimators'], contamination=inputs['contamination'],
                                           random_state=inputs['random state'])
                detector.fit(features)
                outlier = detector.predict(features)

                normal = df[outlier == 1]
                abnormal = df[outlier == -1]
                st.write('**正常样本**')
                st.write(normal)
                tmp_download_link = download_button(normal, f'正常样本.csv', button_text='下载')
                st.markdown(tmp_download_link, unsafe_allow_html=True)

                st.write('**异常样本**')
                st.write(abnormal)
                tmp_download_link = download_button(abnormal, f'异常样本.csv', button_text='下载')
                st.markdown(tmp_download_link, unsafe_allow_html=True)

            elif inputs['model'] == 'DBSCAN':
                model = DBSCAN()
                model.fit(features)
                outlier = model.labels_
                normal = df[outlier == 1]
                abnormal = df[outlier == -1]
                st.write('**正常样本**')
                st.write(normal)
                tmp_download_link = download_button(normal, f'正常样本.csv', button_text='下载')
                st.markdown(tmp_download_link, unsafe_allow_html=True)

                st.write('**异常样本**')
                st.write(abnormal)
                tmp_download_link = download_button(abnormal, f'异常样本.csv', button_text='下载')
                st.markdown(tmp_download_link, unsafe_allow_html=True)

            elif inputs['model'] == 'LocalOutlierFactor':
                detector = LocalOutlierFactor(n_neighbors=inputs['n_neighbors'],
        contamination=inputs['contamination'], metric=inputs['metric'])
                outlier = detector.fit_predict(features)

                normal = df[outlier == 1]
                abnormal = df[outlier == -1]
                st.write('**正常样本**')
                st.write(normal)
                tmp_download_link = download_button(normal, f'正常样本.csv', button_text='下载')
                st.markdown(tmp_download_link, unsafe_allow_html=True)

                st.write('**异常样本**')
                st.write(abnormal)
                tmp_download_link = download_button(abnormal, f'异常样本.csv', button_text='下载')
                st.markdown(tmp_download_link, unsafe_allow_html=True)


elif select_option == "特征工程":
    with st.sidebar:
        sub_option = option_menu(None, ["特征提取", "特征之间的相关性",
                                        "特征与目标的相关性", "特征重要性排序",
                                        ])

    if sub_option == "特征提取":

        colored_header(label="特征提取", description=" ", color_name="blue-90")
        file = st.file_uploader("上传`.csv`文件", type=['csv'],
                                help="只支持csv格式")

        # 创建示例数据
        example_data = pd.DataFrame({
            'Pretty_formula': ['K2SiF6', 'K2LiAlF6']

        })

        # 显示示例数据预览
        with st.expander("📋 查看示例数据格式"):
            st.write("示例数据格式预览：")
            st.dataframe(example_data, use_container_width=True)
            st.caption("请确保您的数据文件包含类似的列结构和数据格式（**PS：运行时电脑剩余虚拟内存需大于100G**）")

        if file is not None:
            colored_header(label="数据信息", description=" ", color_name="blue-70")

            df = pd.read_csv(file)
            df_nrow = df.head()
            st.write(df_nrow)

            button = st.button('🚀 开始提取', type="primary", use_container_width=True)
            if button:
                # 这里会显示进度条
                # df = feature_transform(df)
                # st.write(df.head())
                # tmp_download_link = download_button(df, f'feature_data.csv', button_text='下载')
                # st.markdown(tmp_download_link, unsafe_allow_html=True)
                df = feature_transform(df)

                # 添加文字说明
                st.success("✅ 特征提取完成！")
                st.write("### 特征提取结果预览")
                st.info(f"共生成 {len(df.columns) - 1} 个特征，以下是前5行数据：")

                st.write(df.head())

                st.write("### 数据下载")
                tmp_download_link = download_button(df, f'feature_data.csv', button_text='下载')
                st.markdown(tmp_download_link, unsafe_allow_html=True)

    elif sub_option == "特征之间的相关性":
        colored_header(label="特征之间的相关性", description=" ", color_name="blue-90")
        file = st.file_uploader("上传`.csv`文件", type=['csv'],
                                help="只支持csv格式")

        # 创建示例数据
        example_data = pd.DataFrame({
            '材料编号': ['M001', 'M002', 'M003', 'M004'],
            '特征A': [0.5, 0.6, 0.7, 0.8],
            '特征B': [0.3, 0.25, 0.2, 0.15],
            '特征C': [0.2, 0.15, 0.1, 0.05],
            '发光性能A': [100, 150, 200, 180],
            '发光性能B': [450, 460, 455, 465]
        })

        # 显示示例数据预览
        with st.expander("📋 查看示例数据格式"):
            st.write("示例数据格式预览：")
            st.dataframe(example_data, use_container_width=True)
            st.caption("请确保您的数据文件包含类似的列结构和数据格式")


        if file is not None:
            df = pd.read_csv(file)
            check_string_NaN(df)
            colored_header(label="数据信息", description=" ", color_name="blue-70")
            nrow = st.slider("rows", 1, len(df), 5)
            df_nrow = df.head(nrow)
            st.write(df_nrow)

            colored_header(label="特征与目标", description=" ", color_name="blue-70")

            target_num = st.number_input('目标数量', min_value=1, max_value=10, value=1)

            col_feature, col_target = st.columns(2)
            # features
            features = df.iloc[:, :-target_num]
            # targets
            targets = df.iloc[:, -target_num:]
            with col_feature:
                st.write(features.head())
            with col_target:
                st.write(targets.head())

            colored_header(label="删除高相关特征", description=" ", color_name="blue-30")
            fs = FeatureSelector(features, targets)
            plot = customPlot()

            target_selected_option = st.selectbox('目标', list(fs.targets))
            target_selected = fs.targets[target_selected_option]

            col1, col2 = st.columns([1, 3])
            with col1:
                corr_method = st.selectbox("相关性分析方法", ["pearson", "spearman", "kendall"])
                correlation_threshold = st.slider("相关性阈值", 0.001, 1.0, 0.9)
                corr_matrix = pd.concat([fs.features, target_selected], axis=1).corr(corr_method)
                fs.identify_collinear(corr_matrix, correlation_threshold)
                fs.judge_drop_f_t_after_f_f([target_selected_option], corr_matrix)

                is_mask = st.selectbox('是否隐藏相关矩阵', ('Yes', 'No'))
                with st.expander('**绘图参数**'):
                    options_selected = [plot.set_tick_fontsize(21), plot.set_tick_fontsize(22)]
                with st.expander('高相关特征'):
                    st.write(fs.record_collinear)
            with col2:
                fs.features_dropped_collinear = fs.features.drop(columns=fs.ops['collinear'])
                assert fs.features_dropped_collinear.size != 0, 'zero feature !'
                corr_matrix_drop_collinear = fs.features_dropped_collinear.corr(corr_method)
                plot.corr_cofficient(options_selected, is_mask, corr_matrix_drop_collinear)
                with st.expander('删除后的数据'):
                    data = pd.concat([fs.features_dropped_collinear, targets], axis=1)
                    st.write(data)
                    tmp_download_link = download_button(data, f'dropped_collinear.csv', button_text='下载')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)

    elif sub_option == "特征与目标的相关性":
        colored_header(label="特征与目标的相关性", description=" ", color_name="blue-90")
        file = st.file_uploader("上传`.csv`文件", type=['csv'],
                                help="只支持csv格式")

        # 创建示例数据
        example_data = pd.DataFrame({
            '材料编号': ['M001', 'M002', 'M003', 'M004'],
            '特征A': [0.5, 0.6, 0.7, 0.8],
            '特征B': [0.3, 0.25, 0.2, 0.15],
            '特征C': [0.2, 0.15, 0.1, 0.05],
            '发光性能A': [100, 150, 200, 180],
            '发光性能B': [450, 460, 455, 465]
        })

        # 显示示例数据预览
        with st.expander("📋 查看示例数据格式"):
            st.write("示例数据格式预览：")
            st.dataframe(example_data, use_container_width=True)
            st.caption("请确保您的数据文件包含类似的列结构和数据格式")



        if file is not None:
            df = pd.read_csv(file)
            # 检测缺失值
            check_string_NaN(df)
            colored_header(label="数据信息", description=" ", color_name="blue-70")
            nrow = st.slider("rows", 1, len(df), 5)
            df_nrow = df.head(nrow)
            st.write(df_nrow)

            colored_header(label="特征与目标", description=" ", color_name="blue-70")

            target_num = st.number_input('目标数量', min_value=1, max_value=10, value=1)

            col_feature, col_target = st.columns(2)

            # features
            features = df.iloc[:, :-target_num]
            # targets
            targets = df.iloc[:, -target_num:]
            with col_feature:
                st.write(features.head())
            with col_target:
                st.write(targets.head())

            colored_header(label="删除低相关特征", description=" ", color_name="blue-70")
            fs = FeatureSelector(features, targets)
            plot = customPlot()
            target_selected_option = st.selectbox('feature', list(fs.targets))
            col1, col2 = st.columns([1, 3])

            with col1:
                corr_method = st.selectbox("相关性分析方法", ["pearson", "spearman", "kendall", "MIR"],
                                           key=15)
                if corr_method != "MIR":
                    option_dropped_threshold = st.slider('相关性阈值', 0.0, 1.0, 0.0)
                if corr_method == 'MIR':
                    options_seed = st.checkbox('random state 1024', True)
                with st.expander('**绘图参数**'):
                    options_selected = [plot.set_title_fontsize(11), plot.set_label_fontsize(12),
                                        plot.set_tick_fontsize(13), plot.set_legend_fontsize(14),
                                        plot.set_color('bin color', 0, 10)]

            with col2:
                target_selected = fs.targets[target_selected_option]
                if corr_method != "MIR":
                    corr_matrix = pd.concat([fs.features, target_selected], axis=1).corr(corr_method).abs()

                    fs.judge_drop_f_t([target_selected_option], corr_matrix, option_dropped_threshold)

                    fs.features_dropped_f_t = fs.features.drop(columns=fs.ops['f_t_low_corr'])
                    corr_f_t = pd.concat([fs.features_dropped_f_t, target_selected], axis=1).corr(corr_method)[
                                   target_selected_option][:-1]

                    plot.corr_feature_target(options_selected, corr_f_t)
                    with st.expander('删除后的数据'):
                        data = pd.concat([fs.features_dropped_f_t, targets], axis=1)
                        st.write(data)
                        tmp_download_link = download_button(data, f'droplowcorr.csv', button_text='下载')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)
                else:
                    if options_seed:
                        corr_mir = MIR(fs.features, target_selected, random_state=1024)
                    else:
                        corr_mir = MIR(fs.features, target_selected)
                    corr_mir = pd.DataFrame(corr_mir).set_index(pd.Index(list(fs.features.columns)))
                    corr_mir.rename(columns={0: 'mutual info'}, inplace=True)
                    plot.corr_feature_target_mir(options_selected, corr_mir)
            st.write('---')



    elif sub_option == "特征重要性排序":
        colored_header(label="特征重要性排序", description=" ", color_name="blue-90")
        file = st.file_uploader("上传`.csv`文件", type=['csv'],
                                help="只支持csv格式")

        # 创建示例数据
        example_data = pd.DataFrame({
            '材料编号': ['M001', 'M002', 'M003', 'M004'],
            '特征A': [0.5, 0.6, 0.7, 0.8],
            '特征B': [0.3, 0.25, 0.2, 0.15],
            '特征C': [0.2, 0.15, 0.1, 0.05],
            '发光性能A': [100, 150, 200, 180],
            '发光性能B': [450, 460, 455, 465]
        })

        # 显示示例数据预览
        with st.expander("📋 查看示例数据格式"):
            st.write("示例数据格式预览：")
            st.dataframe(example_data, use_container_width=True)
            st.caption("请确保您的数据文件包含类似的列结构和数据格式")

        if file is not None:
            df = pd.read_csv(file)
            # 检测缺失值
            check_string_NaN(df)
            colored_header(label="数据信息", description=" ", color_name="blue-70")
            nrow = st.slider("rows", 1, len(df), 5)
            df_nrow = df.head(nrow)
            st.write(df_nrow)

            colored_header(label="特征与目标", description=" ", color_name="blue-70")

            target_num = st.number_input('目标数量', min_value=1, max_value=10, value=1)

            col_feature, col_target = st.columns(2)
            # features
            features = df.iloc[:, :-target_num]
            # targets
            targets = df.iloc[:, -target_num:]
            with col_feature:
                st.write(features.head())
            with col_target:
                st.write(targets.head())

            fs = FeatureSelector(features, targets)

            colored_header(label="目标", description=" ", color_name="blue-70")

            target_selected_name = st.selectbox('目标', list(fs.targets)[::-1])

            fs.targets = targets[target_selected_name]

            colored_header(label="选择器", description=" ", color_name="blue-70")

            model_path = './models/feature importance'

            template_alg = model_platform(model_path=model_path)

            colored_header(label="训练", description=" ", color_name="blue-70")

            inputs, col2 = template_alg.show()

            if inputs['model'] == 'LinearRegressor':

                fs.model = LinearR()

                with col2:
                    option_cumulative_importance = st.slider('累积重要性阈值', 0.0, 1.0, 0.95)
                    Embedded_method = st.checkbox('更改交叉验证折数', False)
                    if Embedded_method:
                        cv = st.number_input('CV', 1, 20, 5)
                with st.container():
                    button_train = st.button('训练', use_container_width=True)
                if button_train:
                    fs.LinearRegressor()
                    fs.identify_zero_low_importance(option_cumulative_importance)
                    fs.feature_importance_select_show()
                    if Embedded_method:
                        threshold = fs.cumulative_importance

                        feature_importances = fs.feature_importances.set_index('feature', drop=False)

                        features = []
                        scores = []
                        cumuImportance = []
                        for i in range(1, len(fs.features.columns) + 1):
                            features.append(feature_importances.iloc[:i, 0].values.tolist())
                            X_selected = fs.features[features[-1]]
                            score = CVS(fs.model, X_selected, fs.targets, cv=cv, scoring='r2').mean()

                            cumuImportance.append(feature_importances.loc[features[-1][-1], 'cumulative_importance'])
                            scores.append(score)
                        cumu_importance = np.array(cumuImportance)
                        scores = np.array(scores)
                        with plt.style.context(['nature', 'no-latex']):
                            fig, ax = plt.subplots()
                            ax = plt.plot(cumu_importance, scores, 'o-')
                            plt.xlabel("cumulative feature importance")
                            plt.ylabel("r2")
                            st.pyplot(fig)


            elif inputs['model'] == 'LassoRegressor':
                # 创建Lasso模型，使用用户设置的参数
                fs.model = Lasso(
                    alpha=inputs.get('alpha', 1.0),  # 正则化强度
                    max_iter=inputs.get('max_iter', 1000),  # 最大迭代次数
                    tol=inputs.get('tol', 1e-4),  # 收敛容差
                    random_state=inputs.get('random_state', 42),  # 随机种子
                    fit_intercept=inputs.get('fit_intercept', True)  # 是否拟合截距
                )

                with col2:
                    option_cumulative_importance = st.slider('累积重要性阈值', 0.0, 1.0, 0.95)
                    Embedded_method = st.checkbox('更改交叉验证折数', False)
                    if Embedded_method:
                        cv = st.number_input('CV', 1, 20, 5)
                    else:
                        cv = 5  # 默认5折交叉验证

                with st.container():
                    button_train = st.button('训练', use_container_width=True)

                if button_train:
                    # 使用Lasso进行特征选择
                    fs.LassoRegressor()

                    fs.identify_zero_low_importance(option_cumulative_importance)
                    fs.feature_importance_select_show()

                    if Embedded_method:
                        threshold = fs.cumulative_importance
                        feature_importances = fs.feature_importances.set_index('feature', drop=False)

                        features = []
                        scores = []
                        cumuImportance = []

                        for i in range(1, len(fs.features.columns) + 1):
                            features.append(feature_importances.iloc[:i, 0].values.tolist())
                            X_selected = fs.features[features[-1]]
                            score = CVS(fs.model, X_selected, fs.targets, cv=cv, scoring='r2').mean()

                            cumuImportance.append(feature_importances.loc[features[-1][-1], 'cumulative_importance'])
                            scores.append(score)

                        cumu_importance = np.array(cumuImportance)
                        scores = np.array(scores)

                        with plt.style.context(['nature', 'no-latex']):
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.plot(cumu_importance, scores, 'o-', linewidth=2, markersize=6)
                            plt.xlabel("累积特征重要性")
                            plt.ylabel("R² 得分")
                            plt.title("累积特征重要性与模型性能关系")
                            plt.grid(True, alpha=0.3)
                            st.pyplot(fig)

                            # 显示最佳性能点
                            best_idx = np.argmax(scores)
                            st.info(
                                f"最佳R²得分: {scores[best_idx]:.4f} (在累积重要性 {cumu_importance[best_idx]:.3f} 处)")

            elif inputs['model'] == 'RidgeRegressor':

                # 创建Ridge模型，使用用户设置的参数
                fs.model = Ridge(
                    alpha=inputs.get('alpha', 1.0),  # 正则化强度
                    fit_intercept=inputs.get('fit_intercept', True),  # 是否拟合截距
                    solver=inputs.get('solver', 'auto')  # 求解器
                    # 移除 random_state 参数，因为Ridge不需要
                )

                with col2:
                    option_cumulative_importance = st.slider('累积重要性阈值', 0.0, 1.0, 0.95)
                    Embedded_method = st.checkbox('更改交叉验证折数', False)
                    if Embedded_method:
                        cv = st.number_input('CV', 1, 20, 5)
                    else:
                        cv = 5  # 默认5折交叉验证

                with st.container():
                    button_train = st.button('训练', use_container_width=True)

                if button_train:
                    # 使用Ridge进行特征选择
                    fs.RidgeRegressor()
                    fs.identify_zero_low_importance(option_cumulative_importance)
                    fs.feature_importance_select_show()

                    if Embedded_method:
                        threshold = fs.cumulative_importance
                        feature_importances = fs.feature_importances.set_index('feature', drop=False)

                        features = []
                        scores = []
                        cumuImportance = []

                        for i in range(1, len(fs.features.columns) + 1):
                            features.append(feature_importances.iloc[:i, 0].values.tolist())
                            X_selected = fs.features[features[-1]]
                            score = CVS(fs.model, X_selected, fs.targets, cv=cv, scoring='r2').mean()

                            cumuImportance.append(feature_importances.loc[features[-1][-1], 'cumulative_importance'])
                            scores.append(score)

                        cumu_importance = np.array(cumuImportance)
                        scores = np.array(scores)

                        with plt.style.context(['nature', 'no-latex']):
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.plot(cumu_importance, scores, 'o-', linewidth=2, markersize=6)
                            plt.xlabel("累积特征重要性")
                            plt.ylabel("R² 得分")
                            plt.title("岭回归 - 累积特征重要性与模型性能关系")
                            plt.grid(True, alpha=0.3)
                            st.pyplot(fig)

                            # 显示最佳性能点
                            best_idx = np.argmax(scores)
                            st.info(
                                f"最佳R²得分: {scores[best_idx]:.4f} (在累积重要性 {cumu_importance[best_idx]:.3f} 处)")

                            # 显示当前使用的参数
                            st.write("当前模型参数:")
                            st.write(f"- 正则化强度 (alpha): {inputs.get('alpha', 1.0)}")
                            st.write(f"- 拟合截距: {'是' if inputs.get('fit_intercept', True) else '否'}")
                            st.write(f"- 求解器: {inputs.get('solver', 'auto')}")

            elif inputs['model'] == 'RandomForestRegressor':

                fs.model = RFR(criterion=inputs['criterion'], n_estimators=inputs['nestimators'],
                               random_state=inputs['random state'], max_depth=inputs['max depth'],
                               min_samples_leaf=inputs['min samples leaf'],
                               min_samples_split=inputs['min samples split'],
                               n_jobs=inputs['njobs'])
                with col2:
                    option_cumulative_importance = st.slider('累积重要性阈值', 0.5, 1.0, 0.95)
                    Embedded_method = st.checkbox('更改交叉验证折数', False)
                    if Embedded_method:
                        cv = st.number_input('CV', 1, 20, 5)

                with st.container():
                    button_train = st.button('训练', use_container_width=True)
                if button_train:

                    fs.RandomForestRegressor()

                    fs.identify_zero_low_importance(option_cumulative_importance)
                    fs.feature_importance_select_show()

                    if Embedded_method:

                        threshold = fs.cumulative_importance

                        feature_importances = fs.feature_importances.set_index('feature', drop=False)

                        features = []
                        scores = []
                        cumuImportance = []
                        for i in range(1, len(fs.features.columns) + 1):
                            features.append(feature_importances.iloc[:i, 0].values.tolist())
                            X_selected = fs.features[features[-1]]
                            score = CVS(fs.model, X_selected, fs.targets, cv=cv, scoring='r2').mean()

                            cumuImportance.append(feature_importances.loc[features[-1][-1], 'cumulative_importance'])
                            scores.append(score)
                        cumu_importance = np.array(cumuImportance)
                        scores = np.array(scores)
                        with plt.style.context(['nature', 'no-latex']):
                            fig, ax = plt.subplots()
                            ax = plt.plot(cumu_importance, scores, 'o-')
                            plt.xlabel("cumulative feature importance")
                            plt.ylabel("r2")
                            st.pyplot(fig)

            st.write('---')


elif select_option == "模型建立":

    colored_header(label="模型建立", description=" ", color_name="blue-90")
    file = st.file_uploader("上传`.csv`文件", type=['csv'],
                            help="只支持csv格式")

    # 创建示例数据
    example_data = pd.DataFrame({
        '材料编号': ['M001', 'M002', 'M003', 'M004'],
        '特征A': [0.5, 0.6, 0.7, 0.8],
        '特征B': [0.3, 0.25, 0.2, 0.15],
        '特征C': [0.2, 0.15, 0.1, 0.05],
        '发光性能A': [100, 150, 200, 180],
        '发光性能B': [450, 460, 455, 465]
    })

    # 显示示例数据预览
    with st.expander("📋 查看示例数据格式"):
        st.write("示例数据格式预览：")
        st.dataframe(example_data, use_container_width=True)
        st.caption("请确保您的数据文件包含类似的列结构和数据格式")


    if file is not None:
        df = pd.read_csv(file)
        # 检测缺失值
        check_string_NaN(df)

        colored_header(label="数据信息", description=" ", color_name="blue-70")
        nrow = st.slider("rows", 1, len(df), 5)
        df_nrow = df.head(nrow)
        st.write(df_nrow)

        colored_header(label="特征与目标", description=" ", color_name="blue-70")

        target_num = st.number_input('目标数量', min_value=1, max_value=10, value=1)

        col_feature, col_target = st.columns(2)
        # features
        features = df.iloc[:, :-target_num]
        # targets
        targets = df.iloc[:, -target_num:]
        with col_feature:
            st.write(features.head())
        with col_target:
            st.write(targets.head())
        # =================== model ====================================
        reg = REGRESSOR(features, targets)

        colored_header(label="目标", description=" ", color_name="blue-70")

        target_selected_option = st.selectbox('目标', list(reg.targets)[::-1])

        reg.targets = targets[target_selected_option]

        colored_header(label="回归器", description=" ", color_name="blue-30")

        model_path = './models/regressors'

        template_alg = model_platform(model_path)

        inputs, col2 = template_alg.show()

        if inputs['model'] == 'DecisionTreeRegressor':

            with col2:
                with st.expander('操作配置'):
                    operator = st.selectbox('', ('训练集测试集划分', '交叉验证评估', '留一法验证'),
                                            label_visibility="collapsed")
                    if operator == '训练集测试集划分':
                        inputs['测试集比例'] = st.slider('测试集比例', 0.1, 0.5, 0.2)
                        reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features, reg.targets,
                                                                           test_size=inputs['测试集比例'],
                                                                           random_state=inputs['random state'])

                    elif operator == '交叉验证评估':
                        cv = st.number_input('交叉验证折数', 1, 20, 5)

                    elif operator == '留一法验证':
                        loo = LeaveOneOut()

            colored_header(label="模型训练", description=" ", color_name="blue-30")
            with st.container():
                button_train = st.button('🚀 开始训练', type="primary", use_container_width=True)
            if button_train:
                if operator == '训练集测试集划分':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = tree.DecisionTreeRegressor(random_state=inputs['random state'],
                                                               splitter=inputs['splitter'],
                                                               max_depth=inputs['max depth'],
                                                               min_samples_leaf=inputs['min samples leaf'],
                                                               min_samples_split=inputs['min samples split'])

                        reg.DecisionTreeRegressor()


                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual', 'prediction']
                        plot_and_export_results(reg, "DecisionTreeRegressor")

                        if inputs['tree graph']:
                            class_names = list(set(reg.targets.astype(str).tolist()))
                            dot_data = tree.export_graphviz(reg.model, out_file=None, feature_names=list(reg.features),
                                                            class_names=class_names, filled=True, rounded=True)
                            graph = graphviz.Source(dot_data)
                            graph.render('决策树结构图', view=True)

                    elif inputs['auto hyperparameters']:
                        def DTR_TT(max_depth, min_samples_leaf, min_samples_split):
                            reg.model = tree.DecisionTreeRegressor(max_depth=int(max_depth),
                                                                   min_samples_leaf=int(min_samples_leaf),
                                                                   min_samples_split=int(min_samples_split))
                            reg.DecisionTreeRegressor()
                            return reg.score


                        DTRbounds = {'max_depth': (1, inputs['max depth']),
                                     'min_samples_leaf': (1, inputs['min samples leaf']),
                                     'min_samples_split': (2, inputs['min samples split'])}

                        with st.expander('超参数优化'):

                            optimizer = BayesianOptimization(f=DTR_TT, pbounds=DTRbounds,
                                                             random_state=inputs['random state'],
                                                             allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])

                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['max_depth'] = int(params_best['max_depth'])
                        params_best['min_samples_leaf'] = int(params_best['min_samples_leaf'])
                        params_best['min_samples_split'] = int(params_best['min_samples_split'])
                        st.write("\n", "\n", "最优参数: ", params_best)

                        reg.model = tree.DecisionTreeRegressor(random_state=inputs['random state'],
                                                               splitter=inputs['splitter'],
                                                               max_depth=params_best['max_depth'],
                                                               min_samples_leaf=params_best['min_samples_leaf'],
                                                               min_samples_split=params_best['min_samples_split'])

                        reg.DecisionTreeRegressor()

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual', 'prediction']
                        plot_and_export_results(reg, "DecisionTreeRegressor")

                        if inputs['tree graph']:
                            class_names = list(set(reg.targets.astype(str).tolist()))
                            dot_data = tree.export_graphviz(reg.model, out_file=None, feature_names=list(reg.features),
                                                            class_names=class_names, filled=True, rounded=True)
                            graph = graphviz.Source(dot_data)
                            graph.render('决策树结构图', view=True)

                elif operator == '交叉验证评估':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = tree.DecisionTreeRegressor(random_state=inputs['random state'],
                                                               splitter=inputs['splitter'],
                                                               max_depth=inputs['max depth'],
                                                               min_samples_leaf=inputs['min samples leaf'],
                                                               min_samples_split=inputs['min samples split'])

                        export_cross_val_results(reg, cv, "DecisionTreeRegressor_交叉验证", inputs['random state'])

                    elif inputs['auto hyperparameters']:
                        def DTR_TT(max_depth, min_samples_leaf, min_samples_split):
                            reg.model = tree.DecisionTreeRegressor(max_depth=int(max_depth),
                                                                   min_samples_leaf=int(min_samples_leaf),
                                                                   min_samples_split=int(min_samples_split))
                            cv_score = cv_cal(reg, cv, inputs['random state'])
                            return cv_score


                        DTRbounds = {'max_depth': (1, inputs['max depth']),
                                     'min_samples_leaf': (1, inputs['min samples leaf']),
                                     'min_samples_split': (2, inputs['min samples split'])}

                        with st.expander('超参数优化'):

                            optimizer = BayesianOptimization(f=DTR_TT, pbounds=DTRbounds,
                                                             random_state=inputs['random state'],
                                                             allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])

                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['max_depth'] = int(params_best['max_depth'])
                        params_best['min_samples_leaf'] = int(params_best['min_samples_leaf'])
                        params_best['min_samples_split'] = int(params_best['min_samples_split'])
                        st.write("\n", "\n", "最优参数: ", params_best)
                        reg.model = tree.DecisionTreeRegressor(random_state=inputs['random state'],
                                                               splitter=inputs['splitter'],
                                                               max_depth=params_best['max_depth'],
                                                               min_samples_leaf=params_best['min_samples_leaf'],
                                                               min_samples_split=params_best['min_samples_split'])

                        export_cross_val_results(reg, cv, "DecisionTreeRegressor_交叉验证", inputs['random state'])


                elif operator == '留一法验证':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = tree.DecisionTreeRegressor(random_state=inputs['random state'],
                                                               splitter=inputs['splitter'],
                                                               max_depth=inputs['max depth'],
                                                               min_samples_leaf=inputs['min samples leaf'],
                                                               min_samples_split=inputs['min samples split'])

                        export_loo_results(reg, loo, "DecisionTreeRegressor_留一法")

                    elif inputs['auto hyperparameters']:
                        def DTR_TT(max_depth, min_samples_leaf, min_samples_split):
                            reg.model = tree.DecisionTreeRegressor(max_depth=int(max_depth),
                                                                   min_samples_leaf=int(min_samples_leaf),
                                                                   min_samples_split=int(min_samples_split))
                            loo_score = loo_cal(reg, loo)
                            return loo_score


                        DTRbounds = {'max_depth': (1, inputs['max depth']),
                                     'min_samples_leaf': (1, inputs['min samples leaf']),
                                     'min_samples_split': (2, inputs['min samples split'])}

                        with st.expander('超参数优化'):

                            optimizer = BayesianOptimization(f=DTR_TT, pbounds=DTRbounds,
                                                             random_state=inputs['random state'],
                                                             allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])

                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['max_depth'] = int(params_best['max_depth'])
                        params_best['min_samples_leaf'] = int(params_best['min_samples_leaf'])
                        params_best['min_samples_split'] = int(params_best['min_samples_split'])
                        st.write("\n", "\n", "最优参数: ", params_best)
                        reg.model = tree.DecisionTreeRegressor(random_state=inputs['random state'],
                                                               splitter=inputs['splitter'],
                                                               max_depth=params_best['max_depth'],
                                                               min_samples_leaf=params_best['min_samples_leaf'],
                                                               min_samples_split=params_best['min_samples_split'])
                        export_loo_results(reg, loo, "DecisionTreeRegressor_留一法")

        if inputs['model'] == 'RandomForestRegressor':
            with col2:
                with st.expander('操作配置'):
                    operator = st.selectbox('数据操作方式', ('训练集测试集划分', '交叉验证评估', '留一法验证'))
                    if operator == '训练集测试集划分':
                        inputs['测试集比例'] = st.slider('测试集比例', 0.1, 0.5, 0.2)
                        reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features, reg.targets,
                                                                           test_size=inputs['测试集比例'],
                                                                           random_state=inputs['random state'])
                    elif operator == '交叉验证评估':
                        cv = st.number_input('交叉验证折数', 1, 20, 5)

                    elif operator == '留一法验证':
                        loo = LeaveOneOut()

            colored_header(label="模型训练", description=" ", color_name="blue-30")
            with st.container():
                button_train = st.button('🚀 开始训练', type="primary", use_container_width=True)

            if button_train:
                if operator == '训练集测试集划分':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = RFR(n_estimators=inputs['nestimators'], random_state=inputs['random state'],
                                        max_depth=inputs['max depth'], min_samples_leaf=inputs['min samples leaf'],
                                        min_samples_split=inputs['min samples split'], oob_score=inputs['oob score'],
                                        warm_start=inputs['warm start'],
                                        n_jobs=inputs['njobs'])
                        reg.RandomForestRegressor()

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual', 'prediction']
                        plot_and_export_results(reg, "RandomForestRegressor")

                    elif inputs['auto hyperparameters']:
                        def RFR_TT(n_estimators, max_depth, min_samples_leaf, min_samples_split):

                            reg.model = RFR(n_estimators=int(n_estimators), max_depth=int(max_depth),
                                            min_samples_leaf=int(min_samples_leaf),
                                            min_samples_split=int(min_samples_split), n_jobs=-1)
                            reg.RandomForestRegressor()
                            return reg.score


                        RFRbounds = {'n_estimators': (1, inputs['nestimators']), 'max_depth': (1, inputs['max depth']),
                                     'min_samples_leaf': (1, inputs['min samples leaf']),
                                     'min_samples_split': (2, inputs['min samples split'])}

                        with st.expander('超参数优化'):
                            optimizer = BayesianOptimization(f=RFR_TT, pbounds=RFRbounds,
                                                             random_state=inputs['random state'],
                                                             allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['n_estimators'] = int(params_best['n_estimators'])
                        params_best['max_depth'] = int(params_best['max_depth'])
                        params_best['min_samples_leaf'] = int(params_best['min_samples_leaf'])
                        params_best['min_samples_split'] = int(params_best['min_samples_split'])
                        st.write("\n", "\n", "最优参数: ", params_best)

                        reg.model = RFR(n_estimators=params_best['n_estimators'], random_state=inputs['random state'],
                                        max_depth=params_best['max_depth'],
                                        min_samples_leaf=params_best['min_samples_leaf'],
                                        min_samples_split=params_best['min_samples_split'],
                                        oob_score=inputs['oob score'], warm_start=inputs['warm start'],
                                        n_jobs=inputs['njobs'])

                        reg.RandomForestRegressor()

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual', 'prediction']
                        plot_and_export_results(reg, "RandomForestRegressor")

                elif operator == '交叉验证评估':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = RFR(n_estimators=inputs['nestimators'], random_state=inputs['random state'],
                                        max_depth=inputs['max depth'], min_samples_leaf=inputs['min samples leaf'],
                                        min_samples_split=inputs['min samples split'], oob_score=inputs['oob score'],
                                        warm_start=inputs['warm start'],
                                        n_jobs=inputs['njobs'])
                        export_cross_val_results(reg, cv, "RandomForestRegressor_交叉验证", inputs['random state'])
                    elif inputs['auto hyperparameters']:
                        def RFR_TT(n_estimators, max_depth, min_samples_leaf, min_samples_split):
                            reg.model = RFR(n_estimators=int(n_estimators), max_depth=int(max_depth),
                                            min_samples_leaf=int(min_samples_leaf),
                                            min_samples_split=int(min_samples_split), n_jobs=-1)
                            cv_score = cv_cal(reg, cv, inputs['random state'])
                            return cv_score


                        RFRbounds = {'n_estimators': (1, inputs['nestimators']), 'max_depth': (1, inputs['max depth']),
                                     'min_samples_leaf': (1, inputs['min samples leaf']),
                                     'min_samples_split': (2, inputs['min samples split'])}

                        with st.expander('超参数优化'):
                            optimizer = BayesianOptimization(f=RFR_TT, pbounds=RFRbounds,
                                                             random_state=inputs['random state'],
                                                             allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['n_estimators'] = int(params_best['n_estimators'])
                        params_best['max_depth'] = int(params_best['max_depth'])
                        params_best['min_samples_leaf'] = int(params_best['min_samples_leaf'])
                        params_best['min_samples_split'] = int(params_best['min_samples_split'])
                        st.write("\n", "\n", "最优参数: ", params_best)

                        reg.model = RFR(n_estimators=params_best['n_estimators'], random_state=inputs['random state'],
                                        max_depth=params_best['max_depth'],
                                        min_samples_leaf=params_best['min_samples_leaf'],
                                        min_samples_split=params_best['min_samples_split'],
                                        oob_score=inputs['oob score'], warm_start=inputs['warm start'],
                                        n_jobs=inputs['njobs'])

                        export_cross_val_results(reg, cv, "RandomForestRegressor_交叉验证", inputs['random state'])

                        # elif operator == 'oob score':

                #     reg.model = RFR(criterion = inputs['criterion'],n_estimators=inputs['nestimators'] ,random_state=inputs['random state'],max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
                #                                 min_samples_split=inputs['min samples split'],oob_score=inputs['oob score'], warm_start=inputs['warm start'],
                #                                 n_jobs=inputs['njobs'])

                #     reg_res  = reg.model.fit(reg.features, reg.targets)
                #     oob_score = reg_res.oob_score_
                #     st.write(f'袋外分数 : {oob_score}')

                elif operator == '留一法验证':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = RFR(criterion=inputs['criterion'], n_estimators=inputs['nestimators'],
                                        random_state=inputs['random state'], max_depth=inputs['max depth'],
                                        min_samples_leaf=inputs['min samples leaf'],
                                        min_samples_split=inputs['min samples split'], oob_score=inputs['oob score'],
                                        warm_start=inputs['warm start'],
                                        n_jobs=inputs['njobs'])
                        export_loo_results(reg, loo, "RandomForestRegressor_留一法")
                    elif inputs['auto hyperparameters']:

                        def RFR_TT(n_estimators, max_depth, min_samples_leaf, min_samples_split):
                            reg.model = RFR(n_estimators=int(n_estimators), max_depth=int(max_depth),
                                            min_samples_leaf=int(min_samples_leaf),
                                            min_samples_split=int(min_samples_split), n_jobs=-1)
                            loo_score = loo_cal(reg, loo)
                            return loo_score


                        RFRbounds = {'n_estimators': (1, inputs['nestimators']), 'max_depth': (1, inputs['max depth']),
                                     'min_samples_leaf': (1, inputs['min samples leaf']),
                                     'min_samples_split': (2, inputs['min samples split'])}

                        with st.expander('超参数优化'):
                            optimizer = BayesianOptimization(f=RFR_TT, pbounds=RFRbounds,
                                                             random_state=inputs['random state'],
                                                             allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['n_estimators'] = int(params_best['n_estimators'])
                        params_best['max_depth'] = int(params_best['max_depth'])
                        params_best['min_samples_leaf'] = int(params_best['min_samples_leaf'])
                        params_best['min_samples_split'] = int(params_best['min_samples_split'])
                        st.write("\n", "\n", "最优参数: ", params_best)

                        reg.model = RFR(n_estimators=params_best['n_estimators'], random_state=inputs['random state'],
                                        max_depth=params_best['max_depth'],
                                        min_samples_leaf=params_best['min_samples_leaf'],
                                        min_samples_split=params_best['min_samples_split'],
                                        oob_score=inputs['oob score'], warm_start=inputs['warm start'],
                                        n_jobs=inputs['njobs'])

                        export_loo_results(reg, loo, "RandomForestRegressor_留一法")

        if inputs['model'] == 'SupportVector':

            with col2:
                with st.expander('操作配置'):

                    preprocess = st.selectbox('数据预处理', ['StandardScaler', 'MinMaxScaler'])

                    operator = st.selectbox('operator', ('训练集测试集划分', '交叉验证评估', '留一法验证'),
                                            label_visibility='collapsed')
                    if operator == '训练集测试集划分':
                        inputs['测试集比例'] = st.slider('测试集比例', 0.1, 0.5, 0.2)
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)

                        reg.features = pd.DataFrame(reg.features)

                        reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features, reg.targets,
                                                                           test_size=inputs['测试集比例'],
                                                                           random_state=inputs['random state'])

                    elif operator == '交叉验证评估':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        cv = st.number_input('交叉验证折数', 1, 20, 5)

                    elif operator == '留一法验证':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        loo = LeaveOneOut()
            colored_header(label="模型训练", description=" ", color_name="blue-30")
            with st.container():
                button_train = st.button('🚀 开始训练', type="primary", use_container_width=True)
            if button_train:
                if operator == '训练集测试集划分':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = SVR(kernel=inputs['kernel'], C=inputs['C'])

                        reg.SupportVector()

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual', 'prediction']

                        plot_and_export_results(reg, "SupportVector")

                    elif inputs['auto hyperparameters']:
                        def SVR_TT(C):
                            reg.model = SVR(kernel='rbf', C=C)
                            reg.SupportVector()
                            return reg.score


                        SVRbounds = {'C': (0.001, inputs['C'])}

                        with st.expander('超参数优化'):
                            optimizer = BayesianOptimization(f=SVR_TT, pbounds=SVRbounds,
                                                             random_state=inputs['random state'],
                                                             allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['kernel'] = 'rbf'
                        st.write("\n", "\n", "最优参数: ", params_best)

                        reg.model = SVR(kernel='rbf', C=params_best['C'])

                        reg.SupportVector()

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual', 'prediction']
                        plot_and_export_results(reg, "SupportVector")
                elif operator == '交叉验证评估':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = SVR(kernel=inputs['kernel'], C=inputs['C'])

                        export_cross_val_results(reg, cv, "SupportVector_交叉验证", inputs['random state'])
                    elif inputs['auto hyperparameters']:
                        def SVR_TT(C):
                            reg.model = SVR(kernel='rbf', C=C)
                            cv_score = cv_cal(reg, cv, inputs['random state'])
                            return cv_score


                        SVRbounds = {'C': (0.001, inputs['C'])}

                        with st.expander('超参数优化'):
                            optimizer = BayesianOptimization(f=SVR_TT, pbounds=SVRbounds,
                                                             random_state=inputs['random state'],
                                                             allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['kernel'] = 'rbf'
                        st.write("\n", "\n", "最优参数: ", params_best)

                        reg.model = SVR(kernel='rbf', C=params_best['C'])

                        export_cross_val_results(reg, cv, "SupportVector_交叉验证", inputs['random state'])

                elif operator == '留一法验证':
                    if inputs['auto hyperparameters'] == False:
                        # kernel = PairwiseKernel()
                        reg.model = SVR(kernel=inputs['kernel'], C=inputs['C'])
                        # reg.model = SVR(kernel=kernel, C=inputs['C'])

                        export_loo_results(reg, loo, "SupportVector_留一法")
                    elif inputs['auto hyperparameters']:
                        def SVR_TT(C):
                            reg.model = SVR(kernel='rbf', C=C)
                            loo_score = loo_cal(reg, loo)
                            return loo_score


                        SVRbounds = {'C': (0.001, inputs['C'])}

                        with st.expander('超参数优化'):
                            optimizer = BayesianOptimization(f=SVR_TT, pbounds=SVRbounds,
                                                             random_state=inputs['random state'],
                                                             allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['kernel'] = 'rbf'
                        st.write("\n", "\n", "最优参数: ", params_best)

                        reg.model = SVR(kernel='rbf', C=params_best['C'])

                        export_loo_results(reg, loo, "SupportVector_留一法")

        if inputs['model'] == 'GPRegressor':

            with col2:
                with st.expander('操作配置'):

                    preprocess = st.selectbox('数据预处理', ['StandardScaler', 'MinMaxScaler'])

                    operator = st.selectbox('operator', ('训练集测试集划分', '交叉验证评估', '留一法验证'),
                                            label_visibility='collapsed')
                    if operator == '训练集测试集划分':
                        inputs['测试集比例'] = st.slider('测试集比例', 0.1, 0.5, 0.2)
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)

                        reg.features = pd.DataFrame(reg.features)

                        reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features, reg.targets,
                                                                           test_size=inputs['测试集比例'],
                                                                           random_state=inputs['random state'])

                    elif operator == '交叉验证评估':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        cv = st.number_input('交叉验证折数', 1, 20, 5)

                    elif operator == '留一法验证':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        loo = LeaveOneOut()
            colored_header(label="模型训练", description=" ", color_name="blue-30")
            with st.container():
                button_train = st.button('🚀 开始训练', type="primary", use_container_width=True)
            if button_train:
                if operator == '训练集测试集划分':
                    if inputs['kernel'] == None:
                        kernel = None
                    elif inputs['kernel'] == 'DotProduct':
                        kernel = DotProduct()
                    elif inputs['kernel'] == 'WhiteKernel':
                        kernel = WhiteKernel()
                    elif inputs['kernel'] == 'DotProduct+WhiteKernel':
                        kernel = DotProduct() + WhiteKernel()
                    elif inputs['kernel'] == 'Matern':
                        kernel = Matern()
                    elif inputs['kernel'] == 'PairwiseKernel':
                        kernel = PairwiseKernel()
                    elif inputs['kernel'] == 'RationalQuadratic':
                        kernel = RationalQuadratic()
                    elif inputs['kernel'] == 'RBF':
                        kernel = RBF()
                    elif inputs['kernel'] == 'DotProduct+RationalQuadratic':
                        kernel = DotProduct() + RationalQuadratic()
                    elif inputs['kernel'] == 'PairwiseKernel+RationalQuadratic':
                        kernel = PairwiseKernel() + RationalQuadratic()
                    elif inputs['kernel'] == 'DotProduct+PairwiseKernel':
                        kernel = DotProduct() + PairwiseKernel()

                    reg.model = GPR(kernel=kernel, random_state=inputs['random state'])

                    reg.GPRegressor()

                    result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                    result_data.columns = ['actual', 'prediction']

                    plot_and_export_results(reg, "GPRegressor")

                elif operator == '交叉验证评估':
                    if inputs['kernel'] == None:
                        kernel = None
                    elif inputs['kernel'] == 'DotProduct':
                        kernel = DotProduct()
                    elif inputs['kernel'] == 'WhiteKernel':
                        kernel = WhiteKernel()
                    elif inputs['kernel'] == 'DotProduct+WhiteKernel':
                        kernel = DotProduct() + WhiteKernel()
                    elif inputs['kernel'] == 'Matern':
                        kernel = Matern()
                    elif inputs['kernel'] == 'PairwiseKernel':
                        kernel = PairwiseKernel()
                    elif inputs['kernel'] == 'RationalQuadratic':
                        kernel = RationalQuadratic()
                    elif inputs['kernel'] == 'RBF':
                        kernel = RBF()
                    elif inputs['kernel'] == 'DotProduct+RationalQuadratic':
                        kernel = DotProduct() + RationalQuadratic()
                    elif inputs['kernel'] == 'PairwiseKernel+RationalQuadratic':
                        kernel = PairwiseKernel() + RationalQuadratic()
                    elif inputs['kernel'] == 'DotProduct+PairwiseKernel':
                        kernel = DotProduct() + PairwiseKernel()
                    reg.model = GPR(kernel=kernel, random_state=inputs['random state'])

                    export_cross_val_results(reg, cv, "GPRegressor_交叉验证", inputs['random state'])

                elif operator == '留一法验证':
                    if inputs['kernel'] == None:
                        kernel = None
                    elif inputs['kernel'] == 'DotProduct':
                        kernel = DotProduct()
                    elif inputs['kernel'] == 'WhiteKernel':
                        kernel = WhiteKernel()
                    elif inputs['kernel'] == 'DotProduct+WhiteKernel':
                        kernel = DotProduct() + WhiteKernel()
                    elif inputs['kernel'] == 'Matern':
                        kernel = Matern()
                    elif inputs['kernel'] == 'PairwiseKernel':
                        kernel = PairwiseKernel()
                    elif inputs['kernel'] == 'RationalQuadratic':
                        kernel = RationalQuadratic()
                    elif inputs['kernel'] == 'RBF':
                        kernel = RBF()
                    elif inputs['kernel'] == 'DotProduct+RationalQuadratic':
                        kernel = DotProduct() + RationalQuadratic()
                    elif inputs['kernel'] == 'PairwiseKernel+RationalQuadratic':
                        kernel = PairwiseKernel() + RationalQuadratic()
                    elif inputs['kernel'] == 'DotProduct+PairwiseKernel':
                        kernel = DotProduct() + PairwiseKernel()
                    reg.model = GPR(kernel=kernel, random_state=inputs['random state'])

                    export_loo_results(reg, loo, "GPRegressor_留一法")

        if inputs['model'] == 'KNeighborsRegressor':

            with col2:
                with st.expander('操作配置'):

                    preprocess = st.selectbox('数据预处理', ['StandardScaler', 'MinMaxScaler'])

                    operator = st.selectbox('operator', ('训练集测试集划分', '交叉验证评估', '留一法验证'),
                                            label_visibility='collapsed')
                    if operator == '训练集测试集划分':
                        inputs['测试集比例'] = st.slider('测试集比例', 0.1, 0.5, 0.2)
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)

                        reg.features = pd.DataFrame(reg.features)

                        reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features, reg.targets,
                                                                           test_size=inputs['测试集比例'],
                                                                           random_state=inputs['random state'])

                    elif operator == '交叉验证评估':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        cv = st.number_input('交叉验证折数', 1, 20, 5)

                    elif operator == '留一法验证':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        loo = LeaveOneOut()

            colored_header(label="模型训练", description=" ", color_name="blue-30")
            with st.container():
                button_train = st.button('🚀 开始训练', type="primary", use_container_width=True)
            if button_train:
                if operator == '训练集测试集划分':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = KNeighborsRegressor(n_neighbors=inputs['n neighbors'])

                        reg.KNeighborsRegressor()

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual', 'prediction']

                        plot_and_export_results(reg, "KNeighborsRegressor")
                    elif inputs['auto hyperparameters']:
                        def KNNR_TT(n_neighbors):
                            reg.model = KNeighborsRegressor(n_neighbors=int(n_neighbors))
                            reg.KNeighborsRegressor()
                            return reg.score


                        KNNRbounds = {'n_neighbors': (1, inputs['n neighbors'])}

                        with st.expander('超参数优化'):
                            optimizer = BayesianOptimization(f=KNNR_TT, pbounds=KNNRbounds,
                                                             random_state=inputs['random state'],
                                                             allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['n_neighbors'] = int(params_best['n_neighbors'])
                        st.write("\n", "\n", "最佳参数: ", params_best)

                        reg.model = KNeighborsRegressor(n_neighbors=params_best['n_neighbors'])

                        reg.KNeighborsRegressor()

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual', 'prediction']
                        plot_and_export_results(reg, "KNeighborsRegressor")
                elif operator == '交叉验证评估':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = KNeighborsRegressor(n_neighbors=inputs['n neighbors'])

                        export_cross_val_results(reg, cv, "KNeighborsRegressor_交叉验证", inputs['random state'])
                    elif inputs['auto hyperparameters']:
                        def KNNR_TT(n_neighbors):
                            reg.model = KNeighborsRegressor(n_neighbors=int(n_neighbors))
                            cv_score = cv_cal(reg, cv, inputs['random state'])
                            return cv_score


                        KNNRbounds = {'n_neighbors': (1, inputs['n neighbors'])}

                        with st.expander('超参数优化'):
                            optimizer = BayesianOptimization(f=KNNR_TT, pbounds=KNNRbounds,
                                                             random_state=inputs['random state'],
                                                             allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['n_neighbors'] = int(params_best['n_neighbors'])
                        st.write("\n", "\n", "最佳参数: ", params_best)

                        reg.model = KNeighborsRegressor(n_neighbors=params_best['n_neighbors'])
                        export_cross_val_results(reg, cv, "KNeighborsRegressor_交叉验证", inputs['random state'])

                elif operator == '留一法验证':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = KNeighborsRegressor(n_neighbors=inputs['n neighbors'])

                        export_loo_results(reg, loo, "KNeighborsRegressor_留一法")
                    elif inputs['auto hyperparameters']:
                        def KNNR_TT(n_neighbors):
                            reg.model = KNeighborsRegressor(n_neighbors=int(n_neighbors))
                            loo_score = loo_cal(reg, loo)
                            return loo_score


                        KNNRbounds = {'n_neighbors': (1, inputs['n neighbors'])}

                        with st.expander('超参数优化'):
                            optimizer = BayesianOptimization(f=KNNR_TT, pbounds=KNNRbounds,
                                                             random_state=inputs['random state'],
                                                             allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['n_neighbors'] = int(params_best['n_neighbors'])
                        st.write("\n", "\n", "最佳参数: ", params_best)

                        reg.model = KNeighborsRegressor(n_neighbors=params_best['n_neighbors'])
                        export_loo_results(reg, loo, "KNeighborsRegressor_留一法")

        if inputs['model'] == 'LinearRegressor':

            with col2:
                with st.expander('操作配置'):

                    preprocess = st.selectbox('数据预处理', ['StandardScaler', 'MinMaxScaler'])

                    operator = st.selectbox('operator', ('训练集测试集划分', '交叉验证评估', '留一法验证'),
                                            label_visibility='collapsed')
                    if operator == '训练集测试集划分':
                        inputs['测试集比例'] = st.slider('测试集比例', 0.1, 0.5, 0.2)
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)

                        reg.features = pd.DataFrame(reg.features)

                        reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features, reg.targets,
                                                                           test_size=inputs['测试集比例'],
                                                                           random_state=inputs['random state'])

                    elif operator == '交叉验证评估':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        cv = st.number_input('交叉验证折数', 1, 20, 5)

                    elif operator == '留一法验证':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        loo = LeaveOneOut()

            colored_header(label="模型训练", description=" ", color_name="blue-30")
            with st.container():
                # button_train = st.button('🚀 开始训练', type="primary", use_container_width=True)
                button_train = st.button('🚀 开始训练', type="primary", use_container_width=True)

            if button_train:
                if operator == '训练集测试集划分':

                    reg.model = LinearR()

                    reg.LinearRegressor()

                    result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                    result_data.columns = ['actual', 'prediction']
                    plot_and_export_results(reg, "LinearRegressor")


                elif operator == '交叉验证评估':

                    reg.model = LinearR()

                    export_cross_val_results(reg, cv, "LinearRegressor_交叉验证", inputs['random state'])


                elif operator == '留一法验证':

                    reg.model = LinearR()

                    export_loo_results(reg, loo, "LinearRegressor_留一法")

        if inputs['model'] == 'LassoRegressor':

            with col2:
                with st.expander('操作配置'):

                    preprocess = st.selectbox('数据预处理', ['StandardScaler', 'MinMaxScaler'])

                    operator = st.selectbox('operator', ('训练集测试集划分', '交叉验证评估', '留一法验证'),
                                            label_visibility='collapsed')
                    if operator == '训练集测试集划分':
                        inputs['测试集比例'] = st.slider('测试集比例', 0.1, 0.5, 0.2)
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)

                        reg.features = pd.DataFrame(reg.features)

                        reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features, reg.targets,
                                                                           test_size=inputs['测试集比例'],
                                                                           random_state=inputs['random state'])

                    elif operator == '交叉验证评估':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        cv = st.number_input('交叉验证折数', 1, 20, 5)

                    elif operator == '留一法验证':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        loo = LeaveOneOut()

            colored_header(label="模型训练", description=" ", color_name="blue-30")
            with st.container():
                button_train = st.button('🚀 开始训练', type="primary", use_container_width=True)
            if button_train:
                if operator == '训练集测试集划分':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = Lasso(alpha=inputs['alpha'], random_state=inputs['random state'])

                        reg.LassoRegressor()

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual', 'prediction']

                        plot_and_export_results(reg, "LassoRegressor")

                    elif inputs['auto hyperparameters']:
                        def LassoR_TT(alpha):
                            reg.model = Lasso(alpha=alpha)
                            reg.LassoRegressor()
                            return reg.score


                        LassoRbounds = {'alpha': (0.001, inputs['alpha'])}

                        with st.expander('超参数优化'):
                            optimizer = BayesianOptimization(f=LassoR_TT, pbounds=LassoRbounds,
                                                             random_state=inputs['random state'],
                                                             allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        st.write("\n", "\n", "最佳参数: ", params_best)

                        reg.model = Lasso(alpha=params_best['alpha'], random_state=inputs['random state'])

                        reg.LassoRegressor()

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual', 'prediction']
                        plot_and_export_results(reg, "LassoRegressor")
                elif operator == '交叉验证评估':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = Lasso(alpha=inputs['alpha'], random_state=inputs['random state'])

                        export_cross_val_results(reg, cv, "LassoRegressor_交叉验证", inputs['random state'])

                    elif inputs['auto hyperparameters']:
                        def LassoR_TT(alpha):
                            reg.model = Lasso(alpha=alpha)
                            cv_score = cv_cal(reg, cv, inputs['random state'])
                            return cv_score


                        LassoRbounds = {'alpha': (0.001, inputs['alpha'])}

                        with st.expander('超参数优化'):
                            optimizer = BayesianOptimization(f=LassoR_TT, pbounds=LassoRbounds,
                                                             random_state=inputs['random state'],
                                                             allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        st.write("\n", "\n", "最佳参数: ", params_best)

                        reg.model = Lasso(alpha=params_best['alpha'], random_state=inputs['random state'])
                        export_cross_val_results(reg, cv, "LassoRegressor_交叉验证", inputs['random state'])

                elif operator == '留一法验证':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = Lasso(alpha=inputs['alpha'], random_state=inputs['random state'])

                        export_loo_results(reg, loo, "LassoRegressor_留一法")
                    elif inputs['auto hyperparameters']:
                        def LassoR_TT(alpha):
                            reg.model = Lasso(alpha=alpha)
                            loo_score = loo_cal(reg, loo)
                            return loo_score


                        LassoRbounds = {'alpha': (0.001, inputs['alpha'])}

                        with st.expander('超参数优化'):
                            optimizer = BayesianOptimization(f=LassoR_TT, pbounds=LassoRbounds,
                                                             random_state=inputs['random state'],
                                                             allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        st.write("\n", "\n", "最佳参数: ", params_best)

                        reg.model = Lasso(alpha=params_best['alpha'], random_state=inputs['random state'])
                        export_loo_results(reg, loo, "LassoRegressor_留一法")

        if inputs['model'] == 'RidgeRegressor':
            with col2:
                with st.expander('操作配置'):

                    preprocess = st.selectbox('数据预处理', ['StandardScaler', 'MinMaxScaler'])

                    operator = st.selectbox('operator', ('训练集测试集划分', '交叉验证评估', '留一法验证'))
                    if operator == '训练集测试集划分':
                        inputs['测试集比例'] = st.slider('测试集比例', 0.1, 0.5, 0.2)
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)

                        reg.features = pd.DataFrame(reg.features)

                        reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features, reg.targets,
                                                                           test_size=inputs['测试集比例'],
                                                                           random_state=inputs['random state'])

                    elif operator == '交叉验证评估':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        cv = st.number_input('交叉验证折数', 1, 20, 5)

                    elif operator == '留一法验证':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        loo = LeaveOneOut()

            colored_header(label="模型训练", description=" ", color_name="blue-30")
            with st.container():
                button_train = st.button('🚀 开始训练', type="primary", use_container_width=True)
            if button_train:
                if operator == '训练集测试集划分':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = Ridge(alpha=inputs['alpha'], random_state=inputs['random state'])

                        reg.RidgeRegressor()

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual', 'prediction']

                        plot_and_export_results(reg, "RidgeRegressor")

                    elif inputs['auto hyperparameters']:
                        def RidgeR_TT(alpha):
                            reg.model = Ridge(alpha=alpha)
                            reg.RidgeRegressor()
                            return reg.score


                        RidgeRbounds = {'alpha': (0.001, inputs['alpha'])}

                        with st.expander('超参数优化'):
                            optimizer = BayesianOptimization(f=RidgeR_TT, pbounds=RidgeRbounds,
                                                             random_state=inputs['random state'],
                                                             allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        st.write("\n", "\n", "最佳参数: ", params_best)

                        reg.model = Ridge(alpha=params_best['alpha'], random_state=inputs['random state'])

                        reg.RidgeRegressor()

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual', 'prediction']
                        plot_and_export_results(reg, "RidgeRegressor")
                elif operator == '交叉验证评估':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = Ridge(alpha=inputs['alpha'], random_state=inputs['random state'])

                        export_cross_val_results(reg, cv, "RidgeRegressor_交叉验证", inputs['random state'])
                    elif inputs['auto hyperparameters']:
                        def RidgeR_TT(alpha):
                            reg.model = Ridge(alpha=alpha)
                            cv_score = cv_cal(reg, cv, inputs['random state'])
                            return cv_score


                        RidgeRbounds = {'alpha': (0.001, inputs['alpha'])}

                        with st.expander('超参数优化'):
                            optimizer = BayesianOptimization(f=RidgeR_TT, pbounds=RidgeRbounds,
                                                             random_state=inputs['random state'],
                                                             allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        st.write("\n", "\n", "最佳参数: ", params_best)

                        reg.model = Ridge(alpha=params_best['alpha'], random_state=inputs['random state'])
                        export_cross_val_results(reg, cv, "RidgeRegressor_交叉验证", inputs['random state'])

                elif operator == '留一法验证':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = Ridge(alpha=inputs['alpha'], random_state=inputs['random state'])

                        export_loo_results(reg, loo, "RidgeRegressor_留一法")
                    elif inputs['auto hyperparameters']:
                        def RidgeR_TT(alpha):
                            reg.model = Ridge(alpha=alpha)
                            loo_score = loo_cal(reg, loo)
                            return loo_score


                        RidgeRbounds = {'alpha': (0.001, inputs['alpha'])}

                        with st.expander('超参数优化'):
                            optimizer = BayesianOptimization(f=RidgeR_TT, pbounds=RidgeRbounds,
                                                             random_state=inputs['random state'],
                                                             allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        st.write("\n", "\n", "最佳参数: ", params_best)

                        reg.model = Ridge(alpha=params_best['alpha'], random_state=inputs['random state'])
                        export_loo_results(reg, loo, "RidgeRegressor_留一法")

        if inputs['model'] == 'GradientBoostingRegressor':

            with col2:
                with st.expander('操作配置'):
                    operator = st.selectbox('operator', ('训练集测试集划分', '交叉验证评估', '留一法验证'))
                    if operator == '训练集测试集划分':
                        inputs['测试集比例'] = st.slider('测试集比例', 0.1, 0.5, 0.2)
                        reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features, reg.targets,
                                                                           test_size=inputs['测试集比例'],
                                                                           random_state=inputs['random state'])
                    elif operator == '交叉验证评估':
                        cv = st.number_input('交叉验证折数', 1, 20, 5)
                    elif operator == '留一法验证':
                        loo = LeaveOneOut()

            colored_header(label="模型训练", description=" ", color_name="blue-30")
            with st.container():
                button_train = st.button('🚀 开始训练', type="primary", use_container_width=True)
            if button_train:

                if operator == '训练集测试集划分':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = GradientBoostingRegressor(learning_rate=inputs['learning rate'],
                                                              n_estimators=inputs['nestimators'],
                                                              max_features=inputs['max features'],
                                                              random_state=inputs['random state'])

                        reg.GradientBoostingRegressor()

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual', 'prediction']
                        plot_and_export_results(reg, "GradientBoostingRegressor")

                    elif inputs['auto hyperparameters']:
                        def GBR_TT(learning_rate, n_estimators):
                            reg.model = GradientBoostingRegressor(learning_rate=learning_rate,
                                                                  n_estimators=int(n_estimators),
                                                                  max_features=inputs['max features'])
                            reg.GradientBoostingRegressor()
                            return reg.score


                        GBRbounds = {'learning_rate': (0.001, inputs['learning rate']),
                                     'n_estimators': (1, inputs['nestimators'])}

                        with st.expander('超参数优化'):
                            optimizer = BayesianOptimization(f=GBR_TT, pbounds=GBRbounds,
                                                             random_state=inputs['random state'],
                                                             allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['n_estimators'] = int(params_best['n_estimators'])
                        params_best['max_features'] = inputs['max features']
                        st.write("\n", "\n", "最佳参数: ", params_best)
                        reg.model = GradientBoostingRegressor(learning_rate=params_best['learning_rate'],
                                                              n_estimators=params_best['n_estimators'],
                                                              max_features=params_best['max_features'],
                                                              random_state=inputs['random state'])
                        reg.GradientBoostingRegressor()

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual', 'prediction']
                        plot_and_export_results(reg, "GradientBoostingRegressor")
                elif operator == '交叉验证评估':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = GradientBoostingRegressor(learning_rate=inputs['learning rate'],
                                                              n_estimators=inputs['nestimators'],
                                                              max_features=inputs['max features'],
                                                              random_state=inputs['random state'])

                        export_cross_val_results(reg, cv, "GradientBoostingRegressor_交叉验证", inputs['random state'])
                    elif inputs['auto hyperparameters']:
                        def GBR_TT(learning_rate, n_estimators):
                            reg.model = GradientBoostingRegressor(learning_rate=learning_rate,
                                                                  n_estimators=int(n_estimators),
                                                                  max_features=inputs['max features'])
                            cv_score = cv_cal(reg, cv, inputs['random state'])
                            return cv_score


                        GBRbounds = {'learning_rate': (0.001, inputs['learning rate']),
                                     'n_estimators': (1, inputs['nestimators'])}

                        with st.expander('超参数优化'):
                            optimizer = BayesianOptimization(f=GBR_TT, pbounds=GBRbounds,
                                                             random_state=inputs['random state'],
                                                             allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['n_estimators'] = int(params_best['n_estimators'])
                        params_best['max_features'] = inputs['max features']
                        st.write("\n", "\n", "最佳参数: ", params_best)
                        reg.model = GradientBoostingRegressor(learning_rate=params_best['learning_rate'],
                                                              n_estimators=params_best['n_estimators'],
                                                              max_features=params_best['max_features'],
                                                              random_state=inputs['random state'])
                        export_cross_val_results(reg, cv, "GradientBoostingRegressor_交叉验证", inputs['random state'])

                elif operator == '留一法验证':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = GradientBoostingRegressor(learning_rate=inputs['learning rate'],
                                                              n_estimators=inputs['nestimators'],
                                                              max_features=inputs['max features'],
                                                              random_state=inputs['random state'])
                        export_loo_results(reg, loo, "GradientBoostingRegressor_留一法")
                    elif inputs['auto hyperparameters']:
                        def GBR_TT(learning_rate, n_estimators):
                            reg.model = GradientBoostingRegressor(learning_rate=learning_rate,
                                                                  n_estimators=int(n_estimators),
                                                                  max_features=inputs['max features'])
                            loo_score = loo_cal(reg, loo)
                            return loo_score


                        GBRbounds = {'learning_rate': (0.001, inputs['learning rate']),
                                     'n_estimators': (1, inputs['nestimators'])}

                        with st.expander('超参数优化'):
                            optimizer = BayesianOptimization(f=GBR_TT, pbounds=GBRbounds,
                                                             random_state=inputs['random state'],
                                                             allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['n_estimators'] = int(params_best['n_estimators'])
                        params_best['max_features'] = inputs['max features']
                        st.write("\n", "\n", "最佳参数: ", params_best)
                        reg.model = GradientBoostingRegressor(learning_rate=params_best['learning_rate'],
                                                              n_estimators=params_best['n_estimators'],
                                                              max_features=params_best['max_features'],
                                                              random_state=inputs['random state'])
                        export_loo_results(reg, loo, "GradientBoostingRegressor_留一法")

        if inputs['model'] == 'XGBRegressor':
            with col2:
                with st.expander('操作配置'):
                    preprocess = st.selectbox('数据预处理', ['StandardScaler', 'MinMaxScaler'])
                    operator = st.selectbox('operator', ('训练集测试集划分', '交叉验证评估', '留一法验证'),
                                            label_visibility='collapsed')
                    if operator == '训练集测试集划分':
                        inputs['测试集比例'] = st.slider('测试集比例', 0.1, 0.5, 0.2)
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)

                        reg.features = pd.DataFrame(reg.features)

                        reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features, reg.targets,
                                                                           test_size=inputs['测试集比例'],
                                                                           random_state=inputs['random state'])

                    elif operator == '交叉验证评估':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        cv = st.number_input('交叉验证折数', 1, 20, 5)

                    elif operator == '留一法验证':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        reg.features = pd.DataFrame(reg.features)
                        loo = LeaveOneOut()

            colored_header(label="模型训练", description=" ", color_name="blue-30")
            with st.container():
                button_train = st.button('🚀 开始训练', type="primary", use_container_width=True)
            if button_train:
                if operator == '训练集测试集划分':
                    if inputs['base estimator'] == "gbtree":
                        if inputs['auto hyperparameters'] == False:
                            reg.model = xgb.XGBRegressor(booster=inputs['base estimator'],
                                                         n_estimators=inputs['nestimators'],
                                                         max_depth=inputs['max depth'], subsample=inputs['subsample'],
                                                         colsample_bytree=inputs['subfeature'],
                                                         learning_rate=inputs['learning rate'],
                                                         random_state=inputs['random state'])
                            reg.XGBRegressor()

                            result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                            result_data.columns = ['actual', 'prediction']

                            plot_and_export_results(reg, "XGBRegressor")

                        elif inputs['auto hyperparameters']:
                            def XGBR_TT(n_estimators, max_depth, subsample, colsample_bytree, learning_rate):

                                reg.model = xgb.XGBRegressor(booster=inputs['base estimator'],
                                                             n_estimators=int(n_estimators),
                                                             max_depth=int(max_depth), subsample=subsample,
                                                             colsample_bytree=colsample_bytree,
                                                             learning_rate=learning_rate)
                                reg.XGBRegressor()
                                return reg.score


                            XGBRbounds = {'n_estimators': (1, inputs['nestimators']),
                                          'max_depth': (1, inputs['max depth']),
                                          'subsample': (0.5, inputs['subsample']),
                                          'colsample_bytree': (0.5, inputs['subsample']),
                                          'learning_rate': (0.001, inputs['learning rate'])}

                            with st.expander('超参数优化'):
                                optimizer = BayesianOptimization(f=XGBR_TT, pbounds=XGBRbounds,
                                                                 random_state=inputs['random state'],
                                                                 allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['max_depth'] = int(params_best['max_depth'])
                            params_best['base estimator'] = 'gbtree'
                            st.write("\n", "\n", "最佳参数: ", params_best)

                            reg.model = xgb.XGBRegressor(booster=inputs['base estimator'],
                                                         n_estimators=params_best['n_estimators'],
                                                         max_depth=params_best['max_depth'],
                                                         subsample=params_best['subsample'],
                                                         colsample_bytree=params_best['colsample_bytree'],
                                                         learning_rate=params_best['learning_rate'])

                            reg.XGBRegressor()

                            result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                            result_data.columns = ['actual', 'prediction']
                            plot_and_export_results(reg, "XGBRegressor")

                    elif inputs['base estimator'] == "gblinear":
                        if inputs['auto hyperparameters'] == False:
                            reg.model = xgb.XGBRegressor(booster=inputs['base estimator'],
                                                         n_estimators=inputs['nestimators'],
                                                         max_depth=inputs['max depth'], subsample=inputs['subsample'],
                                                         colsample_bytree=inputs['subfeature'],
                                                         learning_rate=inputs['learning rate'],
                                                         random_state=inputs['random state'])
                            reg.XGBRegressor()

                            result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                            result_data.columns = ['actual', 'prediction']

                            plot_and_export_results(reg, "XGBRegressor")

                        elif inputs['auto hyperparameters']:
                            def XGBR_TT(n_estimators, max_depth, subsample, colsample_bytree, learning_rate):

                                reg.model = xgb.XGBRegressor(booster=inputs['base estimator'],
                                                             n_estimators=int(n_estimators),
                                                             max_depth=int(max_depth), subsample=subsample,
                                                             colsample_bytree=colsample_bytree,
                                                             learning_rate=learning_rate)
                                reg.XGBRegressor()
                                return reg.score


                            XGBRbounds = {'n_estimators': (1, inputs['nestimators']),
                                          'max_depth': (1, inputs['max depth']),
                                          'subsample': (0.5, inputs['subsample']),
                                          'colsample_bytree': (0.5, inputs['subsample']),
                                          'learning_rate': (0.001, inputs['learning rate'])}

                            with st.expander('超参数优化'):
                                optimizer = BayesianOptimization(f=XGBR_TT, pbounds=XGBRbounds,
                                                                 random_state=inputs['random state'],
                                                                 allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['max_depth'] = int(params_best['max_depth'])
                            params_best['base estimator'] = 'gblinear'
                            st.write("\n", "\n", "最佳参数: ", params_best)

                            reg.model = xgb.XGBRegressor(booster=inputs['base estimator'],
                                                         n_estimators=params_best['n_estimators'],
                                                         max_depth=params_best['max_depth'],
                                                         subsample=params_best['subsample'],
                                                         colsample_bytree=params_best['colsample_bytree'],
                                                         learning_rate=params_best['learning_rate'])

                            reg.XGBRegressor()

                            result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                            result_data.columns = ['actual', 'prediction']
                            plot_and_export_results(reg, "XGBRegressor")

                elif operator == '交叉验证评估':
                    if inputs['base estimator'] == "gbtree":
                        if inputs['auto hyperparameters'] == False:
                            reg.model = xgb.XGBRegressor(booster=inputs['base estimator'],
                                                         n_estimators=inputs['nestimators'],
                                                         max_depth=inputs['max depth'], subsample=inputs['subsample'],
                                                         colsample_bytree=inputs['subfeature'],
                                                         learning_rate=inputs['learning rate'],
                                                         random_state=inputs['random state'])

                            cvs = CV(reg.model, reg.features, reg.targets, cv=cv, scoring=make_scorer(r2_score),
                                     return_train_score=False, return_estimator=True)

                            export_cross_val_results(reg, cv, "XGBRegressor_交叉验证", inputs['random state'])
                        elif inputs['auto hyperparameters']:
                            def XGBR_TT(n_estimators, max_depth, subsample, colsample_bytree, learning_rate):

                                reg.model = xgb.XGBRegressor(booster=inputs['base estimator'],
                                                             n_estimators=int(n_estimators),
                                                             max_depth=int(max_depth), subsample=subsample,
                                                             colsample_bytree=colsample_bytree,
                                                             learning_rate=learning_rate)
                                cv_score = cv_cal(reg, cv, inputs['random state'])
                                return cv_score


                            XGBRbounds = {'n_estimators': (1, inputs['nestimators']),
                                          'max_depth': (1, inputs['max depth']),
                                          'subsample': (0.5, inputs['subsample']),
                                          'colsample_bytree': (0.5, inputs['subsample']),
                                          'learning_rate': (0.001, inputs['learning rate'])}

                            with st.expander('超参数优化'):
                                optimizer = BayesianOptimization(f=XGBR_TT, pbounds=XGBRbounds,
                                                                 random_state=inputs['random state'],
                                                                 allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['max_depth'] = int(params_best['max_depth'])
                            params_best['base estimator'] = 'gbtree'
                            st.write("\n", "\n", "最佳参数: ", params_best)

                            reg.model = xgb.XGBRegressor(booster=inputs['base estimator'],
                                                         n_estimators=params_best['n_estimators'],
                                                         max_depth=params_best['max_depth'],
                                                         subsample=params_best['subsample'],
                                                         colsample_bytree=params_best['colsample_bytree'],
                                                         learning_rate=params_best['learning_rate'])

                            export_cross_val_results(reg, cv, "XGBRegressor_交叉验证", inputs['random state'])

                    elif inputs['base estimator'] == "gblinear":
                        if inputs['auto hyperparameters'] == False:
                            reg.model = xgb.XGBRegressor(booster=inputs['base estimator'],
                                                         n_estimators=inputs['nestimators'],
                                                         max_depth=inputs['max depth'], subsample=inputs['subsample'],
                                                         colsample_bytree=inputs['subfeature'],
                                                         learning_rate=inputs['learning rate'],
                                                         random_state=inputs['random state'])
                            # cvs = CV(reg.model, reg.features, reg.targets, cv = cv, scoring=make_scorer(r2_score), return_train_score=False, return_estimator=True)

                            export_cross_val_results(reg, cv, "XGBRegressor_交叉验证", inputs['random state'])
                        elif inputs['auto hyperparameters']:
                            def XGBR_TT(n_estimators, max_depth, subsample, colsample_bytree, learning_rate):

                                reg.model = xgb.XGBRegressor(booster=inputs['base estimator'],
                                                             n_estimators=int(n_estimators),
                                                             max_depth=int(max_depth), subsample=subsample,
                                                             colsample_bytree=colsample_bytree,
                                                             learning_rate=learning_rate)
                                cv_score = cv_cal(reg, cv, inputs['random state'])
                                return cv_score


                            XGBRbounds = {'n_estimators': (1, inputs['nestimators']),
                                          'max_depth': (1, inputs['max depth']),
                                          'subsample': (0.5, inputs['subsample']),
                                          'colsample_bytree': (0.5, inputs['subsample']),
                                          'learning_rate': (0.001, inputs['learning rate'])}

                            with st.expander('超参数优化'):
                                optimizer = BayesianOptimization(f=XGBR_TT, pbounds=XGBRbounds,
                                                                 random_state=inputs['random state'],
                                                                 allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['max_depth'] = int(params_best['max_depth'])
                            params_best['base estimator'] = 'gbtree'
                            st.write("\n", "\n", "最佳参数: ", params_best)

                            reg.model = xgb.XGBRegressor(booster=inputs['base estimator'],
                                                         n_estimators=params_best['n_estimators'],
                                                         max_depth=params_best['max_depth'],
                                                         subsample=params_best['subsample'],
                                                         colsample_bytree=params_best['colsample_bytree'],
                                                         learning_rate=params_best['learning_rate'])

                            export_cross_val_results(reg, cv, "XGBRegressor_交叉验证", inputs['random state'])

                elif operator == '留一法验证':
                    if inputs['base estimator'] == "gbtree":
                        if inputs['auto hyperparameters'] == False:
                            reg.model = xgb.XGBRegressor(booster=inputs['base estimator'],
                                                         n_estimators=inputs['nestimators'],
                                                         max_depth=inputs['max depth'], subsample=inputs['subsample'],
                                                         colsample_bytree=inputs['subfeature'],
                                                         learning_rate=inputs['learning rate'],
                                                         random_state=inputs['random state'])
                            export_loo_results(reg, loo, "XGBRegressor_留一法")
                        elif inputs['auto hyperparameters']:
                            def XGBR_TT(n_estimators, max_depth, subsample, colsample_bytree, learning_rate):

                                reg.model = xgb.XGBRegressor(booster=inputs['base estimator'],
                                                             n_estimators=int(n_estimators),
                                                             max_depth=int(max_depth), subsample=subsample,
                                                             colsample_bytree=colsample_bytree,
                                                             learning_rate=learning_rate)
                                loo_score = loo_cal(reg, loo)
                                return loo_score


                            XGBRbounds = {'n_estimators': (1, inputs['nestimators']),
                                          'max_depth': (1, inputs['max depth']),
                                          'subsample': (0.5, inputs['subsample']),
                                          'colsample_bytree': (0.5, inputs['subsample']),
                                          'learning_rate': (0.001, inputs['learning rate'])}

                            with st.expander('超参数优化'):
                                optimizer = BayesianOptimization(f=XGBR_TT, pbounds=XGBRbounds,
                                                                 random_state=inputs['random state'],
                                                                 allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['max_depth'] = int(params_best['max_depth'])
                            params_best['base estimator'] = 'gbtree'
                            st.write("\n", "\n", "最佳参数: ", params_best)

                            reg.model = xgb.XGBRegressor(booster=inputs['base estimator'],
                                                         n_estimators=params_best['n_estimators'],
                                                         max_depth=params_best['max_depth'],
                                                         subsample=params_best['subsample'],
                                                         colsample_bytree=params_best['colsample_bytree'],
                                                         learning_rate=params_best['learning_rate'])

                            export_loo_results(reg, loo, "XGBRegressor_留一法")

                    elif inputs['base estimator'] == "gblinear":
                        if inputs['auto hyperparameters'] == False:
                            reg.model = xgb.XGBRegressor(booster=inputs['base estimator'],
                                                         n_estimators=inputs['nestimators'],
                                                         max_depth=inputs['max depth'], subsample=inputs['subsample'],
                                                         colsample_bytree=inputs['subfeature'],
                                                         learning_rate=inputs['learning rate'],
                                                         random_state=inputs['random state'])

                            export_loo_results(reg, loo, "XGBRegressor_留一法")

                        elif inputs['auto hyperparameters']:
                            def XGBR_TT(n_estimators, max_depth, subsample, colsample_bytree, learning_rate):

                                reg.model = xgb.XGBRegressor(booster=inputs['base estimator'],
                                                             n_estimators=int(n_estimators),
                                                             max_depth=int(max_depth), subsample=subsample,
                                                             colsample_bytree=colsample_bytree,
                                                             learning_rate=learning_rate)
                                loo_score = loo_cal(reg, loo)
                                return loo_score


                            XGBRbounds = {'n_estimators': (1, inputs['nestimators']),
                                          'max_depth': (1, inputs['max depth']),
                                          'subsample': (0.5, inputs['subsample']),
                                          'colsample_bytree': (0.5, inputs['subsample']),
                                          'learning_rate': (0.001, inputs['learning rate'])}

                            with st.expander('超参数优化'):
                                optimizer = BayesianOptimization(f=XGBR_TT, pbounds=XGBRbounds,
                                                                 random_state=inputs['random state'],
                                                                 allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['max_depth'] = int(params_best['max_depth'])
                            params_best['base estimator'] = 'gbtree'
                            st.write("\n", "\n", "最佳参数: ", params_best)

                            reg.model = xgb.XGBRegressor(booster=inputs['base estimator'],
                                                         n_estimators=params_best['n_estimators'],
                                                         max_depth=params_best['max_depth'],
                                                         subsample=params_best['subsample'],
                                                         colsample_bytree=params_best['colsample_bytree'],
                                                         learning_rate=params_best['learning_rate'])

                            export_loo_results(reg, loo, "XGBRegressor_留一法")

        if inputs['model'] == 'CatBoostRegressor':
            with col2:
                with st.expander('操作配置'):
                    operator = st.selectbox('operator', ('训练集测试集划分', '交叉验证评估', '留一法验证'))

                    if operator == '训练集测试集划分':
                        inputs['测试集比例'] = st.slider('测试集比例', 0.1, 0.5, 0.2)
                        reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features, reg.targets,
                                                                           test_size=inputs['测试集比例'],
                                                                           random_state=inputs['random state'])
                    elif operator == '交叉验证评估':
                        cv = st.number_input('交叉验证折数', 1, 20, 5)
                    elif operator == '留一法验证':
                        loo = LeaveOneOut()
            colored_header(label="模型训练", description=" ", color_name="blue-30")
            with st.container():
                button_train = st.button('🚀 开始训练', type="primary", use_container_width=True)
            if button_train:

                if operator == '训练集测试集划分':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = CatBoostRegressor(iterations=inputs['niteration'],
                                                      learning_rate=inputs['learning rate'], depth=inputs['max depth'],
                                                      random_seed=inputs['random state'])

                        reg.CatBRegressor()

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual', 'prediction']

                        plot_and_export_results(reg, "CatBoostRegressor")
                    elif inputs['auto hyperparameters']:
                        def CatBR_TT(iterations, depth, learning_rate):
                            reg.model = CatBoostRegressor(iterations=int(iterations), learning_rate=learning_rate,
                                                          depth=int(depth))
                            reg.CatBRegressor()
                            return reg.score


                        CatBRbounds = {'iterations': (1, inputs['niteration']), 'depth': (1, inputs['max depth']),
                                       'learning_rate': (0.001, inputs['learning rate'])}

                        with st.expander('超参数优化'):
                            optimizer = BayesianOptimization(f=CatBR_TT, pbounds=CatBRbounds,
                                                             random_state=inputs['random state'],
                                                             allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['iterations'] = int(params_best['iterations'])
                        params_best['depth'] = int(params_best['depth'])
                        st.write("\n", "\n", "最佳参数: ", params_best)

                        reg.model = CatBoostRegressor(iterations=params_best['iterations'],
                                                      learning_rate=params_best['learning_rate'],
                                                      depth=params_best['depth'], random_seed=inputs['random state'])

                        reg.CatBRegressor()

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual', 'prediction']
                        plot_and_export_results(reg, "CatBoostRegressor")
                elif operator == '交叉验证评估':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = CatBoostRegressor(iterations=inputs['niteration'],
                                                      learning_rate=inputs['learning rate'], depth=inputs['max depth'],
                                                      random_seed=inputs['random state'])

                        export_cross_val_results(reg, cv, "CatBoostRegressor_交叉验证", inputs['random state'])
                    elif inputs['auto hyperparameters']:
                        def CatBR_TT(iterations, depth, learning_rate):
                            reg.model = CatBoostRegressor(iterations=int(iterations), learning_rate=learning_rate,
                                                          depth=int(depth))
                            cv_score = cv_cal(reg, cv, inputs['random state'])
                            return cv_score


                        CatBRbounds = {'iterations': (1, inputs['niteration']), 'depth': (1, inputs['max depth']),
                                       'learning_rate': (0.001, inputs['learning rate'])}

                        with st.expander('超参数优化'):
                            optimizer = BayesianOptimization(f=CatBR_TT, pbounds=CatBRbounds,
                                                             random_state=inputs['random state'],
                                                             allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['iterations'] = int(params_best['iterations'])
                        params_best['depth'] = int(params_best['depth'])
                        st.write("\n", "\n", "最佳参数: ", params_best)

                        reg.model = CatBoostRegressor(iterations=params_best['iterations'],
                                                      learning_rate=params_best['learning_rate'],
                                                      depth=params_best['depth'], random_seed=inputs['random state'])
                        export_cross_val_results(reg, cv, "CatBoostRegressor_交叉验证", inputs['random state'])

                elif operator == '留一法验证':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = CatBoostRegressor(iterations=inputs['niteration'],
                                                      learning_rate=inputs['learning rate'], depth=inputs['max depth'],
                                                      random_seed=inputs['random state'])
                        export_loo_results(reg, loo, "CatBoostRegressor_留一法")
                    elif inputs['auto hyperparameters']:
                        def CatBR_TT(iterations, depth, learning_rate):
                            reg.model = CatBoostRegressor(iterations=int(iterations), learning_rate=learning_rate,
                                                          depth=int(depth))
                            loo_score = loo_cal(reg, loo)
                            return loo_score


                        CatBRbounds = {'iterations': (1, inputs['niteration']), 'depth': (1, inputs['max depth']),
                                       'learning_rate': (0.001, inputs['learning rate'])}

                        with st.expander('超参数优化'):
                            optimizer = BayesianOptimization(f=CatBR_TT, pbounds=CatBRbounds,
                                                             random_state=inputs['random state'],
                                                             allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['iterations'] = int(params_best['iterations'])
                        params_best['depth'] = int(params_best['depth'])
                        st.write("\n", "\n", "最佳参数: ", params_best)

                        reg.model = CatBoostRegressor(iterations=params_best['iterations'],
                                                      learning_rate=params_best['learning_rate'],
                                                      depth=params_best['depth'], random_seed=inputs['random state'])
                        export_loo_results(reg, loo, "CatBoostRegressor_留一法")

        if inputs['model'] == 'MLPRegressor':
            with col2:
                with st.expander('操作配置'):

                    preprocess = st.selectbox('数据预处理', ['StandardScaler', 'MinMaxScaler'])

                    operator = st.selectbox('operator', ('训练集测试集划分', '交叉验证评估', '留一法验证'),
                                            label_visibility='collapsed')
                    if operator == '训练集测试集划分':
                        inputs['测试集比例'] = st.slider('测试集比例', 0.1, 0.5, 0.2)
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)

                        reg.features = pd.DataFrame(reg.features)

                        reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features, reg.targets,
                                                                           test_size=inputs['测试集比例'],
                                                                           random_state=inputs['random state'])

                    elif operator == '交叉验证评估':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        cv = st.number_input('交叉验证折数', 1, 20, 5)

                    elif operator == '留一法验证':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        loo = LeaveOneOut()
            colored_header(label="模型训练", description=" ", color_name="blue-30")
            with st.container():
                button_train = st.button('🚀 开始训练', type="primary", use_container_width=True)
            if button_train:
                if operator == '训练集测试集划分':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = MLPRegressor(hidden_layer_sizes=inputs['hidden layer size'],
                                                 activation=inputs['activation'], solver=inputs['solver'],
                                                 batch_size=inputs['batch size'], learning_rate=inputs['learning rate'],
                                                 max_iter=inputs['max iter'],
                                                 random_state=inputs['random state'])
                        reg.MLPRegressor()

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual', 'prediction']
                        plot_and_export_results(reg, "MLPRegressor")
                    elif inputs['auto hyperparameters']:
                        def MLPR_TT(layer_size, neuron_size):
                            layer_size = int(layer_size)
                            neuron_size = int(neuron_size)
                            hidden_layer_size = tuple([neuron_size] * layer_size)
                            reg.model = MLPRegressor(hidden_layer_sizes=hidden_layer_size,
                                                     activation=inputs['activation'], solver=inputs['solver'],
                                                     batch_size=inputs['batch size'],
                                                     learning_rate=inputs['learning rate'], max_iter=inputs['max iter'],
                                                     random_state=inputs['random state'])
                            reg.MLPRegressor()
                            return reg.score


                        MLPRbounds = {'layer_size': (1, inputs['layer size']),
                                      'neuron_size': (1, inputs['neuron size'])}

                        with st.expander('超参数优化'):
                            optimizer = BayesianOptimization(f=MLPR_TT, pbounds=MLPRbounds,
                                                             random_state=inputs['random state'],
                                                             allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['layer_size'] = int(params_best['layer_size'])
                        params_best['neuron_size'] = int(params_best['neuron_size'])
                        st.write("\n", "\n", "最佳参数: ", params_best)
                        hidden_layer_size = tuple(params_best['layer_size'] * [params_best['neuron_size']])
                        reg.model = MLPRegressor(hidden_layer_sizes=hidden_layer_size, activation=inputs['activation'],
                                                 solver=inputs['solver'],
                                                 batch_size=inputs['batch size'], learning_rate=inputs['learning rate'],
                                                 max_iter=inputs['max iter'],
                                                 random_state=inputs['random state'])

                        reg.MLPRegressor()

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual', 'prediction']
                        plot_and_export_results(reg, "MLPRegressor")

                elif operator == '交叉验证评估':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = MLPRegressor(hidden_layer_sizes=inputs['hidden layer size'],
                                                 activation=inputs['activation'], solver=inputs['solver'],
                                                 batch_size=inputs['batch size'], learning_rate=inputs['learning rate'],
                                                 max_iter=inputs['max iter'],
                                                 random_state=inputs['random state'])

                        export_cross_val_results(reg, cv, "MLPRegressor_交叉验证", inputs['random state'])
                    elif inputs['auto hyperparameters']:
                        def MLPR_TT(layer_size, neuron_size):
                            layer_size = int(layer_size)
                            neuron_size = int(neuron_size)
                            hidden_layer_size = tuple([neuron_size] * layer_size)
                            reg.model = MLPRegressor(hidden_layer_sizes=hidden_layer_size,
                                                     activation=inputs['activation'], solver=inputs['solver'],
                                                     batch_size=inputs['batch size'],
                                                     learning_rate=inputs['learning rate'], max_iter=inputs['max iter'],
                                                     random_state=inputs['random state'])
                            cv_score = cv_cal(reg, cv, inputs['random state'])
                            return cv_score


                        MLPRbounds = {'layer_size': (1, inputs['layer size']),
                                      'neuron_size': (1, inputs['neuron size'])}

                        with st.expander('超参数优化'):
                            optimizer = BayesianOptimization(f=MLPR_TT, pbounds=MLPRbounds,
                                                             random_state=inputs['random state'],
                                                             allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['layer_size'] = int(params_best['layer_size'])
                        params_best['neuron_size'] = int(params_best['neuron_size'])
                        st.write("\n", "\n", "最佳参数: ", params_best)
                        hidden_layer_size = tuple(params_best['layer_size'] * [params_best['neuron_size']])
                        reg.model = MLPRegressor(hidden_layer_sizes=hidden_layer_size, activation=inputs['activation'],
                                                 solver=inputs['solver'],
                                                 batch_size=inputs['batch size'], learning_rate=inputs['learning rate'],
                                                 max_iter=inputs['max iter'],
                                                 random_state=inputs['random state'])

                        export_cross_val_results(reg, cv, "MLPRegressor_交叉验证", inputs['random state'])
                elif operator == '留一法验证':
                    if inputs['auto hyperparameters'] == False:
                        reg.model = MLPRegressor(hidden_layer_sizes=inputs['hidden layer size'],
                                                 activation=inputs['activation'], solver=inputs['solver'],
                                                 batch_size=inputs['batch size'], learning_rate=inputs['learning rate'],
                                                 max_iter=inputs['max iter'],
                                                 random_state=inputs['random state'])

                        export_loo_results(reg, loo, "MLPRegressor_留一法")
                    elif inputs['auto hyperparameters']:
                        def MLPR_TT(layer_size, neuron_size):
                            layer_size = int(layer_size)
                            neuron_size = int(neuron_size)
                            hidden_layer_size = tuple([neuron_size] * layer_size)
                            reg.model = MLPRegressor(hidden_layer_sizes=hidden_layer_size,
                                                     activation=inputs['activation'], solver=inputs['solver'],
                                                     batch_size=inputs['batch size'],
                                                     learning_rate=inputs['learning rate'], max_iter=inputs['max iter'],
                                                     random_state=inputs['random state'])
                            loo_score = loo_cal(reg, loo)
                            return loo_score


                        MLPRbounds = {'layer_size': (1, inputs['layer size']),
                                      'neuron_size': (1, inputs['neuron size'])}

                        with st.expander('超参数优化'):
                            optimizer = BayesianOptimization(f=MLPR_TT, pbounds=MLPRbounds,
                                                             random_state=inputs['random state'],
                                                             allow_duplicate_points=True)
                            optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                        params_best = optimizer.max["params"]
                        score_best = optimizer.max["target"]
                        params_best['layer_size'] = int(params_best['layer_size'])
                        params_best['neuron_size'] = int(params_best['neuron_size'])
                        st.write("\n", "\n", "最佳参数: ", params_best)
                        hidden_layer_size = tuple(params_best['layer_size'] * [params_best['neuron_size']])
                        reg.model = MLPRegressor(hidden_layer_sizes=hidden_layer_size, activation=inputs['activation'],
                                                 solver=inputs['solver'],
                                                 batch_size=inputs['batch size'], learning_rate=inputs['learning rate'],
                                                 max_iter=inputs['max iter'],
                                                 random_state=inputs['random state'])

                        export_loo_results(reg, loo, "MLPRegressor_留一法")


        if inputs['model'] == 'BaggingRegressor':
            with col2:
                with st.expander('操作配置'):
                    preprocess = st.selectbox('数据预处理', ['StandardScaler', 'MinMaxScaler'])
                    operator = st.selectbox('operator', ('训练集测试集划分', '交叉验证评估', '留一法验证'),
                                            label_visibility='collapsed')
                    if operator == '训练集测试集划分':
                        inputs['测试集比例'] = st.slider('测试集比例', 0.1, 0.5, 0.2)
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)

                        reg.features = pd.DataFrame(reg.features)

                        reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features, reg.targets,
                                                                           test_size=inputs['测试集比例'],
                                                                           random_state=inputs['random state'])

                    elif operator == '交叉验证评估':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        cv = st.number_input('交叉验证折数', 1, 20, 5)

                    elif operator == '留一法验证':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        reg.features = pd.DataFrame(reg.features)
                        loo = LeaveOneOut()

            colored_header(label="模型训练", description=" ", color_name="blue-30")
            with st.container():
                button_train = st.button('🚀 开始训练', type="primary", use_container_width=True)
            if button_train:

                if operator == '训练集测试集划分':
                    if inputs['base estimator'] == "DecisionTree":
                        if inputs['auto hyperparameters'] == False:
                            reg.model = BaggingRegressor(estimator=None, n_estimators=inputs['nestimators'],
                                                         max_samples=inputs['max samples'],
                                                         max_features=inputs['max features'], n_jobs=-1)

                            reg.BaggingRegressor()

                            result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                            result_data.columns = ['actual', 'prediction']

                            plot_and_export_results(reg, "BaggingR")
                        elif inputs['auto hyperparameters']:
                            def BaggingR_TT(n_estimators, max_samples, max_features):

                                reg.model = BaggingRegressor(estimator=None, n_estimators=int(n_estimators),
                                                             max_samples=int(max_samples),
                                                             max_features=int(max_features), n_jobs=-1)
                                reg.BaggingRegressor()
                                return reg.score


                            BaggingRbounds = {'n_estimators': (1, inputs['nestimators']),
                                              'max_samples': (1, inputs['max samples']),
                                              'max_features': (1, inputs['max features'])}

                            with st.expander('超参数优化'):
                                optimizer = BayesianOptimization(f=BaggingR_TT, pbounds=BaggingRbounds,
                                                                 random_state=inputs['random state'],
                                                                 allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['max_samples'] = int(params_best['max_samples'])
                            params_best['max_features'] = int(params_best['max_features'])
                            params_best['base estimator'] = 'decision tree'
                            st.write("\n", "\n", "最佳参数: ", params_best)

                            reg.model = BaggingRegressor(estimator=None, n_estimators=params_best['n_estimators'],
                                                         max_samples=params_best['max_samples'],
                                                         max_features=params_best['max_features'], n_jobs=-1)

                            reg.BaggingRegressor()

                            result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                            result_data.columns = ['actual', 'prediction']
                            plot_and_export_results(reg, "BaggingR")

                    elif inputs['base estimator'] == "SupportVector":
                        if inputs['auto hyperparameters'] == False:
                            reg.model = BaggingRegressor(estimator=SVR(), n_estimators=inputs['nestimators'],
                                                         max_samples=inputs['max samples'],
                                                         max_features=inputs['max features'], n_jobs=-1)
                            reg.BaggingRegressor()

                            result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                            result_data.columns = ['actual', 'prediction']

                            plot_and_export_results(reg, "BaggingR")
                        elif inputs['auto hyperparameters']:
                            def BaggingR_TT(n_estimators, max_samples, max_features):

                                reg.model = BaggingRegressor(estimator=SVR(), n_estimators=int(n_estimators),
                                                             max_samples=int(max_samples),
                                                             max_features=int(max_features), n_jobs=-1)
                                reg.BaggingRegressor()
                                return reg.score


                            BaggingRbounds = {'n_estimators': (1, inputs['nestimators']),
                                              'max_samples': (1, inputs['max samples']),
                                              'max_features': (1, inputs['max features'])}

                            with st.expander('超参数优化'):
                                optimizer = BayesianOptimization(f=BaggingR_TT, pbounds=BaggingRbounds,
                                                                 random_state=inputs['random state'],
                                                                 allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['max_samples'] = int(params_best['max_samples'])
                            params_best['max_features'] = int(params_best['max_features'])
                            params_best['base estimator'] = 'support vector machine'
                            st.write("\n", "\n", "最佳参数: ", params_best)

                            reg.model = BaggingRegressor(estimator=SVR(), n_estimators=params_best['n_estimators'],
                                                         max_samples=params_best['max_samples'],
                                                         max_features=params_best['max_features'], n_jobs=-1)

                            reg.BaggingRegressor()

                            result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                            result_data.columns = ['actual', 'prediction']
                            plot_and_export_results(reg, "BaggingR")

                    elif inputs['base estimator'] == "LinearRegression":
                        if inputs['auto hyperparameters'] == False:
                            reg.model = BaggingRegressor(estimator=LinearR(), n_estimators=inputs['nestimators'],
                                                         max_samples=inputs['max samples'],
                                                         max_features=inputs['max features'], n_jobs=-1)
                            reg.BaggingRegressor()

                            result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                            result_data.columns = ['actual', 'prediction']

                            plot_and_export_results(reg, "BaggingR")
                        elif inputs['auto hyperparameters']:
                            def BaggingR_TT(n_estimators, max_samples, max_features):

                                reg.model = BaggingRegressor(estimator=LinearR(), n_estimators=int(n_estimators),
                                                             max_samples=int(max_samples),
                                                             max_features=int(max_features), n_jobs=-1)
                                reg.BaggingRegressor()
                                return reg.score


                            BaggingRbounds = {'n_estimators': (1, inputs['nestimators']),
                                              'max_samples': (1, inputs['max samples']),
                                              'max_features': (1, inputs['max features'])}

                            with st.expander('超参数优化'):
                                optimizer = BayesianOptimization(f=BaggingR_TT, pbounds=BaggingRbounds,
                                                                 random_state=inputs['random state'],
                                                                 allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['max_samples'] = int(params_best['max_samples'])
                            params_best['max_features'] = int(params_best['max_features'])
                            params_best['base estimator'] = 'linear regression'
                            st.write("\n", "\n", "最佳参数: ", params_best)

                            reg.model = BaggingRegressor(estimator=LinearR(), n_estimators=params_best['n_estimators'],
                                                         max_samples=params_best['max_samples'],
                                                         max_features=params_best['max_features'], n_jobs=-1)

                            reg.BaggingRegressor()

                            result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                            result_data.columns = ['actual', 'prediction']
                            plot_and_export_results(reg, "BaggingR")

                elif operator == '交叉验证评估':
                    if inputs['base estimator'] == "DecisionTree":
                        if inputs['auto hyperparameters'] == False:
                            reg.model = BaggingRegressor(estimator=None, n_estimators=inputs['nestimators'],
                                                         max_samples=inputs['max samples'],
                                                         max_features=inputs['max features'], n_jobs=-1)
                            # cvs = CV(reg.model, reg.features, reg.targets, cv = cv, scoring=make_scorer(r2_score), return_train_score=False, return_estimator=True)

                            export_cross_val_results(reg, cv, "BaggingR_交叉验证", inputs['random state'])
                        elif inputs['auto hyperparameters']:
                            def BaggingR_TT(n_estimators, max_samples, max_features):

                                reg.model = BaggingRegressor(estimator=None, n_estimators=int(n_estimators),
                                                             max_samples=int(max_samples),
                                                             max_features=int(max_features), n_jobs=-1)
                                cv_score = cv_cal(reg, cv, inputs['random state'])
                                return cv_score


                            BaggingRbounds = {'n_estimators': (1, inputs['nestimators']),
                                              'max_samples': (1, inputs['max samples']),
                                              'max_features': (1, inputs['max features'])}

                            with st.expander('超参数优化'):
                                optimizer = BayesianOptimization(f=BaggingR_TT, pbounds=BaggingRbounds,
                                                                 random_state=inputs['random state'],
                                                                 allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['max_samples'] = int(params_best['max_samples'])
                            params_best['max_features'] = int(params_best['max_features'])
                            params_best['base estimator'] = 'decision tree'
                            st.write("\n", "\n", "最佳参数: ", params_best)

                            reg.model = BaggingRegressor(estimator=None, n_estimators=params_best['n_estimators'],
                                                         max_samples=params_best['max_samples'],
                                                         max_features=params_best['max_features'], n_jobs=-1)

                            export_cross_val_results(reg, cv, "BaggingR_交叉验证", inputs['random state'])

                    elif inputs['base estimator'] == "SupportVector":
                        if inputs['auto hyperparameters'] == False:
                            reg.model = BaggingRegressor(estimator=SVR(), n_estimators=inputs['nestimators'],
                                                         max_samples=inputs['max samples'],
                                                         max_features=inputs['max features'], n_jobs=-1)

                            # cvs = CV(reg.model, reg.features, reg.targets, cv = cv, scoring=make_scorer(r2_score), return_train_score=False, return_estimator=True)

                            export_cross_val_results(reg, cv, "BaggingR_交叉验证", inputs['random state'])

                        elif inputs['auto hyperparameters']:
                            def BaggingR_TT(n_estimators, max_samples, max_features):

                                reg.model = BaggingRegressor(estimator=SVR(), n_estimators=int(n_estimators),
                                                             max_samples=int(max_samples),
                                                             max_features=int(max_features), n_jobs=-1)
                                cv_score = cv_cal(reg, cv, inputs['random state'])
                                return cv_score


                            BaggingRbounds = {'n_estimators': (1, inputs['nestimators']),
                                              'max_samples': (1, inputs['max samples']),
                                              'max_features': (1, inputs['max features'])}

                            with st.expander('超参数优化'):
                                optimizer = BayesianOptimization(f=BaggingR_TT, pbounds=BaggingRbounds,
                                                                 random_state=inputs['random state'],
                                                                 allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['max_samples'] = int(params_best['max_samples'])
                            params_best['max_features'] = int(params_best['max_features'])
                            params_best['base estimator'] = 'support vector machine'
                            st.write("\n", "\n", "最佳特征: ", params_best)

                            reg.model = BaggingRegressor(estimator=None, n_estimators=params_best['n_estimators'],
                                                         max_samples=params_best['max_samples'],
                                                         max_features=params_best['max_features'], n_jobs=-1)

                            export_cross_val_results(reg, cv, "BaggingR_交叉验证", inputs['random state'])

                    elif inputs['base estimator'] == "LinearRegression":
                        if inputs['auto hyperparameters'] == False:
                            reg.model = BaggingRegressor(estimator=LinearR(), n_estimators=inputs['nestimators'],
                                                         max_samples=inputs['max samples'],
                                                         max_features=inputs['max features'], n_jobs=-1)

                            # cvs = CV(reg.model, reg.features, reg.targets, cv = cv, scoring=make_scorer(r2_score), return_train_score=False, return_estimator=True)

                            export_cross_val_results(reg, cv, "BaggingR_交叉验证", inputs['random state'])
                        elif inputs['auto hyperparameters']:
                            def BaggingR_TT(n_estimators, max_samples, max_features):
                                reg.model = BaggingRegressor(estimator=LinearR(), n_estimators=int(n_estimators),
                                                             max_samples=int(max_samples),
                                                             max_features=int(max_features), n_jobs=-1)
                                cv_score = cv_cal(reg, cv, inputs['random state'])
                                return cv_score


                            BaggingRbounds = {'n_estimators': (1, inputs['nestimators']),
                                              'max_samples': (1, inputs['max samples']),
                                              'max_features': (1, inputs['max features'])}

                            with st.expander('超参数优化'):
                                optimizer = BayesianOptimization(f=BaggingR_TT, pbounds=BaggingRbounds,
                                                                 random_state=inputs['random state'],
                                                                 allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['max_samples'] = int(params_best['max_samples'])
                            params_best['max_features'] = int(params_best['max_features'])
                            params_best['base estimator'] = 'linear regression'
                            st.write("\n", "\n", "最佳参数: ", params_best)
                            reg.model = BaggingRegressor(estimator=LinearR(), n_estimators=params_best['n_estimators'],
                                                         max_samples=params_best['max_samples'],
                                                         max_features=params_best['max_features'], n_jobs=-1)

                            export_cross_val_results(reg, cv, "BaggingR_交叉验证", inputs['random state'])
                elif operator == '留一法验证':
                    if inputs['base estimator'] == "DecisionTree":
                        if inputs['auto hyperparameters'] == False:
                            reg.model = BaggingRegressor(
                                estimator=tree.DecisionTreeRegressor(random_state=inputs['random state']),
                                n_estimators=inputs['nestimators'],
                                max_samples=inputs['max samples'], max_features=inputs['max features'], n_jobs=-1)

                            export_loo_results(reg, loo, "BaggingRegressor_留一法")
                        elif inputs['auto hyperparameters']:

                            def BaggingR_TT(n_estimators, max_samples, max_features):
                                reg.model = BaggingRegressor(estimator=None, n_estimators=int(n_estimators),
                                                             max_samples=int(max_samples),
                                                             max_features=int(max_features), n_jobs=-1)
                                loo_score = loo_cal(reg, loo)
                                return loo_score


                            BaggingRbounds = {'n_estimators': (1, inputs['nestimators']),
                                              'max_samples': (1, inputs['max samples']),
                                              'max_features': (1, inputs['max features'])}

                            with st.expander('超参数优化'):
                                optimizer = BayesianOptimization(f=BaggingR_TT, pbounds=BaggingRbounds,
                                                                 random_state=inputs['random state'],
                                                                 allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['max_samples'] = int(params_best['max_samples'])
                            params_best['max_features'] = int(params_best['max_features'])
                            params_best['base estimator'] = 'decision tree'
                            st.write("\n", "\n", "最佳参数: ", params_best)
                            reg.model = BaggingRegressor(estimator=None, n_estimators=params_best['n_estimators'],
                                                         max_samples=params_best['max_samples'],
                                                         max_features=params_best['max_features'], n_jobs=-1)
                            export_loo_results(reg, loo, "BaggingRegressor_留一法")

                    elif inputs['base estimator'] == "SupportVector":
                        if inputs['auto hyperparameters'] == False:
                            reg.model = BaggingRegressor(estimator=SVR(), n_estimators=inputs['nestimators'],
                                                         max_samples=inputs['max samples'],
                                                         max_features=inputs['max features'], n_jobs=-1)
                            export_loo_results(reg, loo, "BaggingRegressor_留一法")
                        elif inputs['auto hyperparameters']:
                            def BaggingR_TT(n_estimators, max_samples, max_features):
                                reg.model = BaggingRegressor(estimator=SVR(), n_estimators=int(n_estimators),
                                                             max_samples=int(max_samples),
                                                             max_features=int(max_features), n_jobs=-1)
                                loo_score = loo_cal(reg, loo)
                                return loo_score


                            BaggingRbounds = {'n_estimators': (1, inputs['nestimators']),
                                              'max_samples': (1, inputs['max samples']),
                                              'max_features': (1, inputs['max features'])}

                            with st.expander('超参数优化'):
                                optimizer = BayesianOptimization(f=BaggingR_TT, pbounds=BaggingRbounds,
                                                                 random_state=inputs['random state'],
                                                                 allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['max_samples'] = int(params_best['max_samples'])
                            params_best['max_features'] = int(params_best['max_features'])
                            params_best['base estimator'] = 'support vector machine'
                            st.write("\n", "\n", "最佳参数: ", params_best)
                            reg.model = BaggingRegressor(estimator=SVR(), n_estimators=params_best['n_estimators'],
                                                         max_samples=params_best['max_samples'],
                                                         max_features=params_best['max_features'], n_jobs=-1)
                            export_loo_results(reg, loo, "BaggingRegressor_留一法")

                    elif inputs['base estimator'] == "LinearRegression":
                        if inputs['auto hyperparameters'] == False:
                            reg.model = BaggingRegressor(estimator=LinearR(), n_estimators=inputs['nestimators'],
                                                         max_samples=inputs['max samples'],
                                                         max_features=inputs['max features'], n_jobs=-1)

                            export_loo_results(reg, loo, "BaggingRegressor_留一法")
                        elif inputs['auto hyperparameters']:
                            def BaggingR_TT(n_estimators, max_samples, max_features):
                                reg.model = BaggingRegressor(estimator=LinearR(), n_estimators=int(n_estimators),
                                                             max_samples=int(max_samples),
                                                             max_features=int(max_features), n_jobs=-1)
                                loo_score = loo_cal(reg, loo)
                                return loo_score


                            BaggingRbounds = {'n_estimators': (1, inputs['nestimators']),
                                              'max_samples': (1, inputs['max samples']),
                                              'max_features': (1, inputs['max features'])}

                            with st.expander('超参数优化'):
                                optimizer = BayesianOptimization(f=BaggingR_TT, pbounds=BaggingRbounds,
                                                                 random_state=inputs['random state'],
                                                                 allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['max_samples'] = int(params_best['max_samples'])
                            params_best['max_features'] = int(params_best['max_features'])
                            params_best['base estimator'] = 'support vector machine'
                            st.write("\n", "\n", "最佳参数: ", params_best)
                            reg.model = BaggingRegressor(estimator=LinearR(), n_estimators=params_best['n_estimators'],
                                                         max_samples=params_best['max_samples'],
                                                         max_features=params_best['max_features'], n_jobs=-1)
                            export_loo_results(reg, loo, "BaggingRegressor_留一法")

        if inputs['model'] == 'AdaBoostRegressor':
            with col2:
                with st.expander('操作配置'):
                    preprocess = st.selectbox('数据预处理', ['StandardScaler', 'MinMaxScaler'])
                    operator = st.selectbox('operator', ('训练集测试集划分', '交叉验证评估', '留一法验证'),
                                            label_visibility='collapsed')
                    if operator == '训练集测试集划分':
                        inputs['测试集比例'] = st.slider('测试集比例', 0.1, 0.5, 0.2)
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)

                        reg.features = pd.DataFrame(reg.features)

                        reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features, reg.targets,
                                                                           test_size=inputs['测试集比例'],
                                                                           random_state=inputs['random state'])
                    elif operator == '交叉验证评估':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        cv = st.number_input('交叉验证折数', 1, 20, 5)

                    elif operator == '留一法验证':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        reg.features = pd.DataFrame(reg.features)
                        loo = LeaveOneOut()

            colored_header(label="模型训练", description=" ", color_name="blue-30")
            with st.container():
                button_train = st.button('🚀 开始训练', type="primary", use_container_width=True)
            if button_train:

                if operator == '训练集测试集划分':

                    if inputs['base estimator'] == "DecisionTree":
                        if inputs['auto hyperparameters'] == False:
                            reg.model = AdaBoostRegressor(estimator=tree.DecisionTreeRegressor(),
                                                          n_estimators=inputs['nestimators'],
                                                          learning_rate=inputs['learning rate'],
                                                          random_state=inputs['random state'])
                            reg.AdaBoostRegressor()
                            result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                            result_data.columns = ['actual', 'prediction']
                            plot_and_export_results(reg, "AdaBoostR")

                        elif inputs['auto hyperparameters']:
                            def AdaBoostR_TT(n_estimators, learning_rate):
                                reg.model = AdaBoostRegressor(estimator=tree.DecisionTreeRegressor(),
                                                              n_estimators=int(n_estimators),
                                                              learning_rate=learning_rate)
                                reg.AdaBoostRegressor()
                                return reg.score


                            AdaBoostRbounds = {'n_estimators': (1, inputs['nestimators']),
                                               'learning_rate': (1, inputs['learning rate'])}

                            with st.expander('超参数优化'):
                                optimizer = BayesianOptimization(f=AdaBoostR_TT, pbounds=AdaBoostRbounds,
                                                                 random_state=inputs['random state'],
                                                                 allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['base estimator'] = 'decision tree'
                            st.write("\n", "\n", "最佳参数: ", params_best)

                            reg.model = AdaBoostRegressor(estimator=tree.DecisionTreeRegressor(),
                                                          n_estimators=params_best['n_estimators'],
                                                          learning_rate=params_best['learning_rate'],
                                                          random_state=inputs['random state'])
                            reg.AdaBoostRegressor()
                            result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                            result_data.columns = ['actual', 'prediction']
                            plot_and_export_results(reg, "AdaBoostR")

                    elif inputs['base estimator'] == "SupportVector":
                        if inputs['auto hyperparameters'] == False:
                            reg.model = AdaBoostRegressor(estimator=SVR(), n_estimators=inputs['nestimators'],
                                                          learning_rate=inputs['learning rate'],
                                                          random_state=inputs['random state'])
                            reg.AdaBoostRegressor()
                            result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                            result_data.columns = ['actual', 'prediction']
                            plot_and_export_results(reg, "AdaBoostR")
                        elif inputs['auto hyperparameters']:
                            def AdaBoostR_TT(n_estimators, learning_rate):
                                reg.model = AdaBoostRegressor(estimator=SVR(),
                                                              n_estimators=int(n_estimators),
                                                              learning_rate=learning_rate)
                                reg.AdaBoostRegressor()
                                return reg.score


                            AdaBoostRbounds = {'n_estimators': (1, inputs['nestimators']),
                                               'learning_rate': (1, inputs['learning rate'])}

                            with st.expander('超参数优化'):
                                optimizer = BayesianOptimization(f=AdaBoostR_TT, pbounds=AdaBoostRbounds,
                                                                 random_state=inputs['random state'],
                                                                 allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['base estimator'] = 'support vector machine'
                            st.write("\n", "\n", "最佳参数: ", params_best)

                            reg.model = AdaBoostRegressor(estimator=SVR(),
                                                          n_estimators=params_best['n_estimators'],
                                                          learning_rate=params_best['learning_rate'],
                                                          random_state=inputs['random state'])
                            reg.AdaBoostRegressor()
                            result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                            result_data.columns = ['actual', 'prediction']
                            plot_and_export_results(reg, "AdaBoostR")


                    elif inputs['base estimator'] == "LinearRegression":
                        if inputs['auto hyperparameters'] == False:
                            reg.model = AdaBoostRegressor(estimator=LinearR(), n_estimators=inputs['nestimators'],
                                                          learning_rate=inputs['learning rate'],
                                                          random_state=inputs['random state'])
                            reg.AdaBoostRegressor()
                            result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                            result_data.columns = ['actual', 'prediction']
                            plot_and_export_results(reg, "AdaBoostR")
                        elif inputs['auto hyperparameters']:
                            def AdaBoostR_TT(n_estimators, learning_rate):
                                reg.model = AdaBoostRegressor(estimator=LinearR(),
                                                              n_estimators=int(n_estimators),
                                                              learning_rate=learning_rate)
                                reg.AdaBoostRegressor()
                                return reg.score


                            AdaBoostRbounds = {'n_estimators': (1, inputs['nestimators']),
                                               'learning_rate': (1, inputs['learning rate'])}

                            with st.expander('超参数优化'):
                                optimizer = BayesianOptimization(f=AdaBoostR_TT, pbounds=AdaBoostRbounds,
                                                                 random_state=inputs['random state'],
                                                                 allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['base estimator'] = 'linear regression'
                            st.write("\n", "\n", "最佳参数: ", params_best)

                            reg.model = AdaBoostRegressor(estimator=LinearR(),
                                                          n_estimators=params_best['n_estimators'],
                                                          learning_rate=params_best['learning_rate'],
                                                          random_state=inputs['random state'])
                            reg.AdaBoostRegressor()
                            result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                            result_data.columns = ['actual', 'prediction']
                            plot_and_export_results(reg, "AdaBoostR")

                elif operator == '交叉验证评估':
                    if inputs['base estimator'] == "DecisionTree":
                        if inputs['auto hyperparameters'] == False:
                            reg.model = AdaBoostRegressor(estimator=tree.DecisionTreeRegressor(),
                                                          n_estimators=inputs['nestimators'],
                                                          learning_rate=inputs['learning rate'],
                                                          random_state=inputs['random state'])
                            # cvs = CV(reg.model, reg.features, reg.targets, cv = cv, scoring=make_scorer(r2_score), return_train_score=False, return_estimator=True)
                            export_cross_val_results(reg, cv, "AdaBoostRegressor_交叉验证", inputs['random state'])
                        elif inputs['auto hyperparameters']:
                            def AdaBoostR_TT(n_estimators, learning_rate):
                                reg.model = AdaBoostRegressor(estimator=tree.DecisionTreeRegressor(),
                                                              n_estimators=int(n_estimators),
                                                              learning_rate=learning_rate)
                                cv_score = cv_cal(reg, cv, inputs['random state'])
                                return cv_score


                            AdaBoostRbounds = {'n_estimators': (1, inputs['nestimators']),
                                               'learning_rate': (1, inputs['learning rate'])}

                            with st.expander('超参数优化'):
                                optimizer = BayesianOptimization(f=AdaBoostR_TT, pbounds=AdaBoostRbounds,
                                                                 random_state=inputs['random state'],
                                                                 allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['base estimator'] = 'decision tree'
                            st.write("\n", "\n", "最佳参数: ", params_best)

                            reg.model = AdaBoostRegressor(estimator=tree.DecisionTreeRegressor(),
                                                          n_estimators=params_best['n_estimators'],
                                                          learning_rate=params_best['learning_rate'],
                                                          random_state=inputs['random state'])

                            export_cross_val_results(reg, cv, "AdaBoostRegressor_交叉验证", inputs['random state'])


                    elif inputs['base estimator'] == "SupportVector":
                        if inputs['auto hyperparameters'] == False:
                            reg.model = AdaBoostRegressor(estimator=SVR(), n_estimators=inputs['nestimators'],
                                                          learning_rate=inputs['learning rate'],
                                                          random_state=inputs['random state'])

                            # cvs = CV(reg.model, reg.features, reg.targets, cv = cv, scoring=make_scorer(r2_score), return_train_score=False, return_estimator=True)

                            export_cross_val_results(reg, cv, "AdaBoostRegressor_交叉验证", inputs['random state'])
                        elif inputs['auto hyperparameters']:
                            def AdaBoostR_TT(n_estimators, learning_rate):
                                reg.model = AdaBoostRegressor(estimator=SVR(),
                                                              n_estimators=int(n_estimators),
                                                              learning_rate=learning_rate)
                                cv_score = cv_cal(reg, cv, inputs['random state'])
                                return cv_score


                            AdaBoostRbounds = {'n_estimators': (1, inputs['nestimators']),
                                               'learning_rate': (1, inputs['learning rate'])}

                            with st.expander('超参数优化'):
                                optimizer = BayesianOptimization(f=AdaBoostR_TT, pbounds=AdaBoostRbounds,
                                                                 random_state=inputs['random state'],
                                                                 allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['base estimator'] = 'support vector machine'
                            st.write("\n", "\n", "最佳参数: ", params_best)

                            reg.model = AdaBoostRegressor(estimator=SVR(),
                                                          n_estimators=params_best['n_estimators'],
                                                          learning_rate=params_best['learning_rate'],
                                                          random_state=inputs['random state'])

                            export_cross_val_results(reg, cv, "AdaBoostRegressor_交叉验证", inputs['random state'])

                    elif inputs['base estimator'] == "LinearRegression":
                        if inputs['auto hyperparameters'] == False:
                            reg.model = reg.model = AdaBoostRegressor(estimator=LinearR(),
                                                                      n_estimators=inputs['nestimators'],
                                                                      learning_rate=inputs['learning rate'],
                                                                      random_state=inputs['random state'])
                            # cvs = CV(reg.model, reg.features, reg.targets, cv = cv, scoring=make_scorer(r2_score), return_train_score=False, return_estimator=True)

                            export_cross_val_results(reg, cv, "AdaBoostRegressor_交叉验证", inputs['random state'])
                        elif inputs['auto hyperparameters']:
                            def AdaBoostR_TT(n_estimators, learning_rate):
                                reg.model = AdaBoostRegressor(estimator=LinearR(),
                                                              n_estimators=int(n_estimators),
                                                              learning_rate=learning_rate)
                                cv_score = cv_cal(reg, cv, inputs['random state'])
                                return cv_score


                            AdaBoostRbounds = {'n_estimators': (1, inputs['nestimators']),
                                               'learning_rate': (1, inputs['learning rate'])}

                            with st.expander('超参数优化'):
                                optimizer = BayesianOptimization(f=AdaBoostR_TT, pbounds=AdaBoostRbounds,
                                                                 random_state=inputs['random state'],
                                                                 allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['base estimator'] = 'linear regression'
                            st.write("\n", "\n", "最佳参数: ", params_best)

                            reg.model = AdaBoostRegressor(estimator=LinearR(),
                                                          n_estimators=params_best['n_estimators'],
                                                          learning_rate=params_best['learning_rate'],
                                                          random_state=inputs['random state'])

                            export_cross_val_results(reg, cv, "AdaBoostRegressor_交叉验证", inputs['random state'])
                elif operator == '留一法验证':
                    if inputs['base estimator'] == "DecisionTree":
                        if inputs['auto hyperparameters'] == False:
                            reg.model = AdaBoostRegressor(estimator=tree.DecisionTreeRegressor(),
                                                          n_estimators=inputs['nestimators'],
                                                          learning_rate=inputs['learning rate'],
                                                          random_state=inputs['random state'])

                            export_loo_results(reg, loo, "AdaBoostRegressor_留一法")
                        elif inputs['auto hyperparameters']:
                            def AdaBoostR_TT(n_estimators, learning_rate):
                                reg.model = AdaBoostRegressor(estimator=tree.DecisionTreeRegressor(),
                                                              n_estimators=int(n_estimators),
                                                              learning_rate=learning_rate)
                                loo_score = loo_cal(reg, loo)
                                return loo_score


                            AdaBoostRbounds = {'n_estimators': (1, inputs['nestimators']),
                                               'learning_rate': (1, inputs['learning rate'])}

                            with st.expander('超参数优化'):
                                optimizer = BayesianOptimization(f=AdaBoostR_TT, pbounds=AdaBoostRbounds,
                                                                 random_state=inputs['random state'],
                                                                 allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['base estimator'] = 'decision tree'
                            st.write("\n", "\n", "最佳参数: ", params_best)

                            reg.model = AdaBoostRegressor(estimator=tree.DecisionTreeRegressor(),
                                                          n_estimators=params_best['n_estimators'],
                                                          learning_rate=params_best['learning_rate'],
                                                          random_state=inputs['random state'])

                            export_loo_results(reg, loo, "AdaBoostRegressor_留一法")

                    elif inputs['base estimator'] == "SupportVector":
                        if inputs['auto hyperparameters'] == False:
                            reg.model = AdaBoostRegressor(estimator=SVR(), n_estimators=inputs['nestimators'],
                                                          learning_rate=inputs['learning rate'],
                                                          random_state=inputs['random state'])

                            export_loo_results(reg, loo, "AdaBoostRegressor_留一法")
                        elif inputs['auto hyperparameters']:
                            def AdaBoostR_TT(n_estimators, learning_rate):
                                reg.model = AdaBoostRegressor(estimator=SVR(),
                                                              n_estimators=int(n_estimators),
                                                              learning_rate=learning_rate)
                                loo_score = loo_cal(reg, loo)
                                return loo_score


                            AdaBoostRbounds = {'n_estimators': (1, inputs['nestimators']),
                                               'learning_rate': (1, inputs['learning rate'])}

                            with st.expander('超参数优化'):
                                optimizer = BayesianOptimization(f=AdaBoostR_TT, pbounds=AdaBoostRbounds,
                                                                 random_state=inputs['random state'],
                                                                 allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['base estimator'] = 'support vector machine'
                            st.write("\n", "\n", "最佳参数: ", params_best)

                            reg.model = AdaBoostRegressor(estimator=SVR(),
                                                          n_estimators=params_best['n_estimators'],
                                                          learning_rate=params_best['learning_rate'],
                                                          random_state=inputs['random state'])

                            export_loo_results(reg, loo, "AdaBoostRegressor_留一法")

                    elif inputs['base estimator'] == "LinearRegression":
                        if inputs['auto hyperparameters'] == False:
                            reg.model = AdaBoostRegressor(estimator=LinearR(), n_estimators=inputs['nestimators'],
                                                          learning_rate=inputs['learning rate'],
                                                          random_state=inputs['random state'])

                            export_loo_results(reg, loo, "AdaBoostRegressor_留一法")
                        elif inputs['auto hyperparameters']:
                            def AdaBoostR_TT(n_estimators, learning_rate):
                                reg.model = AdaBoostRegressor(estimator=LinearR(),
                                                              n_estimators=int(n_estimators),
                                                              learning_rate=learning_rate)
                                loo_score = loo_cal(reg, loo)
                                return loo_score


                            AdaBoostRbounds = {'n_estimators': (1, inputs['nestimators']),
                                               'learning_rate': (1, inputs['learning rate'])}

                            with st.expander('超参数优化'):
                                optimizer = BayesianOptimization(f=AdaBoostR_TT, pbounds=AdaBoostRbounds,
                                                                 random_state=inputs['random state'],
                                                                 allow_duplicate_points=True)
                                optimizer.maximize(init_points=inputs['init points'], n_iter=inputs['iteration number'])
                            params_best = optimizer.max["params"]
                            score_best = optimizer.max["target"]
                            params_best['n_estimators'] = int(params_best['n_estimators'])
                            params_best['base estimator'] = 'linear regression'
                            st.write("\n", "\n", "最佳参数: ", params_best)

                            reg.model = AdaBoostRegressor(estimator=LinearR(),
                                                          n_estimators=params_best['n_estimators'],
                                                          learning_rate=params_best['learning_rate'],
                                                          random_state=inputs['random state'])

                            export_loo_results(reg, loo, "AdaBoostRegressor_留一法")


elif select_option == "AI自动化建模":
        colored_header(label="AI自动化建模", description=" ", color_name="blue-90")
        file = st.file_uploader("上传`.csv`文件", type=['csv'], label_visibility="collapsed")



elif select_option == "SHAP-模型可解释工具":
    colored_header(label="SHAP-模型可解释工具", description=" ", color_name="blue-90")

    # 添加图片
    st.image("界面图片/shap_header.png",
             # caption="SHAP (SHapley Additive exPlanations) 模型可解释性工具",
             width=1000)

    # 使用st.expander来创建可折叠的包围容器
    with st.expander("📁 数据上传区域", expanded=True):
        st.markdown("""
            <div style="text-align: center; 
                        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
                        padding: 15px; 
                        border-radius: 8px;
                        margin: 15px 0;">
                <p style="color: #0d47a1; 
                         font-size: 19px; 
                         font-weight: bold;
                         margin: 0;">
                    请上传CSV格式的数据文件
                </p>
            </div>
            """, unsafe_allow_html=True)

        file = st.file_uploader(
            "选择文件",
            type=['csv'],
            help="文件要求: CSV格式 | 特征列在前 | 目标变量列在后"
            # label_visibility="collapsed"
        )

        if file is None:
            st.caption("等待文件上传...")
        else:
            st.success("文件已上传")

    # 创建示例数据
    example_data = pd.DataFrame({
        '材料编号': ['M001', 'M002', 'M003', 'M004'],
        '特征A': [0.5, 0.6, 0.7, 0.8],
        '特征B': [0.3, 0.25, 0.2, 0.15],
        '特征C': [0.2, 0.15, 0.1, 0.05],
        '发光性能A': [100, 150, 200, 180],
        '发光性能B': [450, 460, 455, 465]
    })

    # 显示示例数据预览
    with st.expander("📋 查看示例数据格式"):
        st.write("示例数据格式预览：")
        st.dataframe(example_data, use_container_width=True)
        st.caption("请确保您的数据文件包含类似的列结构和数据格式")

    if file is not None:
        df = pd.read_csv(file)
        check_string_NaN(df)
        colored_header(label="数据信息", description=" ", color_name="blue-70")
        nrow = st.slider("显示行数", 1, len(df), 5, key="display_rows_slider")
        df_nrow = df.head(nrow)
        st.write(df_nrow)

        colored_header(label="特征与目标变量", description=" ", color_name="blue-70")

        target_num = st.number_input('目标变量数量', min_value=1, max_value=10, value=1, key="target_num_input")

        col_feature, col_target = st.columns(2)

        features = df.iloc[:, :-target_num]
        # targets
        targets = df.iloc[:, -target_num:]
        with col_feature:
            st.write(features.head())
        with col_target:
            st.write(targets.head())

        colored_header(label="SHAP值分析", description=" ", color_name="blue-70")

        fs = FeatureSelector(features, targets)

        target_selected_option = st.selectbox('选择目标变量', list(fs.targets), key="target_selectbox")
        fs.targets = fs.targets[target_selected_option]

        # ==================== 模型选择区域 ====================
        colored_header(label="模型选择", description=" ", color_name="blue-60")

        model_option = st.selectbox(
            '选择机器学习模型',
            ['随机森林回归', '梯度提升回归', '线性回归'],
            help="选择不同的模型进行SHAP分析",
            key="model_selectbox"
        )

        # 根据选择的模型初始化相应的回归器
        if model_option == '随机森林回归':
            from sklearn.ensemble import RandomForestRegressor

            reg = RandomForestRegressor(random_state=42)
            model_name = "随机森林回归"

        elif model_option == '梯度提升回归':
            from sklearn.ensemble import GradientBoostingRegressor

            reg = GradientBoostingRegressor(random_state=42)
            model_name = "梯度提升回归"

        elif model_option == '线性回归':
            from sklearn.linear_model import LinearRegression

            reg = LinearRegression()
            model_name = "线性回归"


        # 显示当前选择的模型
        st.info(f"当前选择模型: **{model_name}**")
        # ==================== 模型选择结束 ====================

        test_size = st.slider('测试集比例', 0.1, 0.5, 0.2, key="test_size_slider")
        # ==================== 可调节的随机种子 ====================
        col1, col2 = st.columns(2)
        with col1:
            use_random_state = st.checkbox('更改随机种子', True, key="use_random_state_checkbox")
        with col2:
            if use_random_state:
                random_seed = st.number_input('随机种子值', min_value=0, max_value=1000, value=42,
                                              key="random_seed_input")
                random_state = random_seed
            else:
                random_state = None
                st.info("默认随机种子（42）")
        # ==================== 随机种子设置结束 ====================

        fs.Xtrain, fs.Xtest, fs.Ytrain, fs.Ytest = TTS(fs.features, fs.targets, test_size=test_size,
                                                       random_state=random_state)

        # 训练模型
        with st.spinner(f'训练{model_name}模型中...'):
            reg.fit(fs.Xtrain, fs.Ytrain)
            train_score = reg.score(fs.Xtrain, fs.Ytrain)
            test_score = reg.score(fs.Xtest, fs.Ytest)

        # 显示模型性能 - 移除key参数
        col1, col2 = st.columns(2)
        with col1:
            st.metric("训练集R²分数", f"{train_score:.4f}")
        with col2:
            st.metric("测试集R²分数", f"{test_score:.4f}")

        # 根据模型类型选择合适的SHAP解释器
        if model_option in ['随机森林回归', '梯度提升回归']:
            explainer = shap.TreeExplainer(reg)
            explainer_type = "TreeExplainer"
        else:
            explainer = shap.Explainer(reg, fs.Xtrain)
            explainer_type = "GeneralExplainer"

        st.info(f"使用 {explainer_type} 进行SHAP分析")
        shap_values = explainer(fs.features)

        colored_header(label="SHAP特征重要性", description=" ", color_name="blue-30")
        nfeatures = st.slider("特征数量", 2, fs.features.shape[1], fs.features.shape[1],
                              key="feature_importance_slider")
        st_shap(shap.plots.bar(shap_values, max_display=nfeatures))

        colored_header(label="SHAP特征聚类", description=" ", color_name="blue-30")
        clustering = shap.utils.hclust(fs.features, fs.targets)
        clustering_cutoff = st.slider('聚类截断阈值', 0.0, 1.0, 0.5, key="clustering_cutoff_slider")
        nfeatures_cluster = st.slider("特征数量", 2, fs.features.shape[1], fs.features.shape[1],
                                      key="feature_cluster_slider")
        st_shap(shap.plots.bar(shap_values, clustering=clustering, clustering_cutoff=clustering_cutoff,
                               max_display=nfeatures_cluster))

        colored_header(label="SHAP蜂群图", description=" ", color_name="blue-30")
        rank_option = st.selectbox('排序方式', ['最大值', '均值'], key="rank_option_selectbox")
        max_display = st.slider('最大显示特征数', 2, fs.features.shape[1], fs.features.shape[1],
                                key="max_display_slider")
        if rank_option == '最大值':
            st_shap(shap.plots.beeswarm(shap_values, order=shap_values.abs.max(0), max_display=max_display))
        else:
            st_shap(shap.plots.beeswarm(shap_values, order=shap_values.abs.mean(0), max_display=max_display))

        colored_header(label="SHAP依赖图", description=" ", color_name="blue-30")

        shap_values_dep = explainer.shap_values(fs.features)
        list_features = fs.features.columns.tolist()
        feature = st.selectbox('选择特征', list_features, key="feature_selectbox")
        interact_feature = st.selectbox('交互特征', list_features, key="interact_feature_selectbox")
        st_shap(shap.dependence_plot(feature, shap_values_dep, fs.features, display_features=fs.features,
                                     interaction_index=interact_feature))



# 页脚
st.markdown("---")
st.caption("无机发光材料AI辅助设计平台 © 2025 | by稀土国家工程研究中心田壮涛")
