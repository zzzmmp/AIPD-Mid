import streamlit as st

# Define possible models in a dict
# Format of the dict: model name -> model code

MODEL = {
    "model": "IsolationForest",
}


def show():
    """Shows the components for the template and returns user inputs as dict."""

    inputs = {}  # dict to store all user inputs until return
    inputs["model"] = MODEL["model"]

    col1, col2 = st.columns([2, 2])

    with col1:
        st.subheader("Isolation Forest 超参数设置")

        # 核心参数
        inputs['n_estimators'] = st.slider(
            "树的数量",
            min_value=1,
            max_value=500,
            value=100,
            step=1,
            help="隔离树的数量，数量越多结果越稳定但计算量越大"
        )

        inputs['contamination'] = st.slider(
            "异常值比例",
            min_value=0.01,
            max_value=0.5,
            value=0.1,
            step=0.01,
            help="数据集中异常值的预期比例"
        )

        # 随机种子
        inputs['random state'] = st.number_input(
            "随机种子",
            min_value=0,
            max_value=100,
            value=42,
            help="设置随机种子以确保结果可重现"
        )

    with col2:
        st.subheader("参数说明")
        st.info("""
        **Isolation Forest 参数指南:**

        - **树的数量**: 建议100-200之间，平衡精度和效率
        - **异常值比例**: 根据数据特性调整，通常0.05-0.2之间
        - **随机种子**: 固定随机性，确保结果可重现

        **建议设置:**
        - 从默认值开始，逐步调整
        - 异常值比例可根据业务经验设定
        """)

        # 显示当前选择的参数
        st.write("当前参数设置:")
        st.write(f"- 树的数量: {inputs['n_estimators']}")
        st.write(f"- 异常值比例: {inputs['contamination']}")
        st.write(f"- 随机种子: {inputs['random state']}")

    return inputs, col2


# To test the alg independent of the app or template, just run
# `streamlit run alg.py` from within this folder.
if __name__ == "__main__":
    show()