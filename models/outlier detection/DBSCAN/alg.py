import streamlit as st

# Define possible models in a dict
# Format of the dict: model name -> model code

MODEL = {
    "model": "DBSCAN",
}


def show():
    """Shows the components for the template and returns user inputs as dict."""

    inputs = {}  # dict to store all user inputs until return

    inputs["model"] = MODEL["model"]

    col1, col2 = st.columns([2, 2])

    with col1:
        st.subheader("DBSCAN 超参数设置")

        # 核心参数
        inputs["eps"] = st.slider(
            "邻域半径 (eps)",
            min_value=0.1,
            max_value=5.0,
            value=0.5,
            step=0.1,
            help="两个样本之间的最大距离，超过此距离则不被视为邻居"
        )

        inputs["min_samples"] = st.number_input(
            "最小样本数",
            min_value=1,
            max_value=20,
            value=5,
            help="形成一个核心点所需的邻域内最小样本数"
        )

        # 可选参数
        inputs["metric"] = st.selectbox(
            "距离度量方法",
            ["欧氏距离", "曼哈顿距离"],
            index=0,
            help="计算样本间距离时使用的度量方法"
        )

    with col2:
        st.subheader("参数说明")
        st.info("""
        **DBSCAN 参数指南:**

        - **eps**: 控制簇的密度，值越小簇越密集
        - **min_samples**: 值越大对噪声越敏感
        - **建议**: 从默认值开始，根据聚类效果调整
        """)

        # 显示当前选择的参数
        st.write("当前参数设置:")
        st.write(f"- 邻域半径: {inputs['eps']}")
        st.write(f"- 最小样本数: {inputs['min_samples']}")
        st.write(f"- 距离度量: {inputs['metric']}")

    return inputs, col2


# To test the alg independent of the app or template, just run
# `streamlit run alg.py` from within this folder.
if __name__ == "__main__":
    show()