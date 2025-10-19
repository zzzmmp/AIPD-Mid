import streamlit as st

# Define possible models in a dict
# Format of the dict: model name -> model code

MODEL = {
    "model": "LocalOutlierFactor",
}


def show():
    """Shows the components for the template and returns user inputs as dict."""

    inputs = {}  # dict to store all user inputs until return
    inputs["model"] = MODEL["model"]

    col1, col2 = st.columns([2, 2])

    with col1:
        st.subheader("LOF 超参数设置")

        # 核心参数
        inputs["n_neighbors"] = st.slider(
            "邻居数量",
            min_value=5,
            max_value=50,
            value=20,
            help="用于计算局部密度的邻居数量"
        )

        inputs["contamination"] = st.slider(
            "异常值比例",
            min_value=0.01,
            max_value=0.5,
            value=0.1,
            step=0.01,
            help="数据集中异常值的预期比例"
        )

        # 算法参数
        inputs["metric"] = st.selectbox(
            "距离度量方法",
            ["euclidean", "manhattan", "minkowski"],
            index=0,
            help="计算样本间距离时使用的度量方法"
        )

    with col2:
        st.subheader("参数说明")
        st.info("""
        **LOF 参数指南:**

        - **邻居数量**: 影响局部密度计算的精度，通常20左右
        - **异常值比例**: 根据数据特性调整，通常0.05-0.2之间
        - **距离度量**: 根据数据分布选择合适的距离计算方法

        **建议设置:**
        - 邻居数量: 10-30之间
        - 异常值比例: 0.05-0.15之间
        """)

        # 显示当前选择的参数
        st.write("当前参数设置:")
        st.write(f"- 邻居数量: {inputs['n_neighbors']}")
        st.write(f"- 异常值比例: {inputs['contamination']}")
        st.write(f"- 距离度量: {inputs['metric']}")

    return inputs, col2


# To test the alg independent of the app or template, just run
# `streamlit run alg.py` from within this folder.
if __name__ == "__main__":
    show()