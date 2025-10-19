import streamlit as st

# Define possible models in a dict
# Format of the dict: model name -> model code

MODEL = {
    "model": "One Class SVM",
}


def show():
    """Shows the components for the template and returns user inputs as dict."""

    inputs = {}  # dict to store all user inputs until return
    inputs["model"] = MODEL["model"]

    col1, col2 = st.columns([2, 2])

    with col1:
        st.subheader("One Class SVM 超参数设置")

        # 核心参数
        inputs['nu'] = st.slider(
            "异常值比例 (nu)",
            min_value=0.01,
            max_value=1.0,
            value=0.05,
            step=0.01,
            help="异常值的上限比例和支持向量的下限比例"
        )

        inputs['gamma'] = st.slider(
            "核系数 (gamma)",
            min_value=0.1,
            max_value=10.0,
            value=0.1,
            step=0.1,
            help="RBF核函数的系数，控制决策边界的形状"
        )

        # 核函数选择
        inputs['kernel'] = st.selectbox(
            "核函数",
            ["rbf", "linear", "poly", "sigmoid"],
            index=0,
            help="用于映射数据的核函数类型"
        )

    with col2:
        st.subheader("参数说明")
        st.info("""
        **One Class SVM 参数指南:**

        - **nu**: 控制异常值比例，值越小对异常值越敏感
        - **gamma**: 控制决策边界复杂度，值越大边界越复杂
        - **核函数**: RBF适用于非线性问题，linear适用于线性问题

        **建议设置:**
        - nu: 0.05-0.2之间
        - gamma: 0.1-1.0之间
        - 通常使用RBF核函数
        """)

        # 显示当前选择的参数
        st.write("当前参数设置:")
        st.write(f"- 异常值比例: {inputs['nu']}")
        st.write(f"- 核系数: {inputs['gamma']}")
        st.write(f"- 核函数: {inputs['kernel']}")

    return inputs, col2


# To test the alg independent of the app or template, just run
# `streamlit run alg.py` from within this folder.
if __name__ == "__main__":
    show()