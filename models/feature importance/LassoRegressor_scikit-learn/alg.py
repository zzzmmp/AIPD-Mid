import streamlit as st

MODEL = {
    "model": "LassoRegressor",
}


def show():
    """Shows the components for the template and returns user inputs as dict."""

    inputs = {}  # dict to store all user inputs until return
    inputs["model"] = MODEL["model"]

    col1, col2 = st.columns([2, 2])

    with col1:
        st.subheader("Lasso回归 超参数设置")

        # 核心参数
        inputs['alpha'] = st.slider(
            "正则化强度 (alpha)",
            min_value=0.01,
            max_value=10.0,
            value=1.0,
            step=0.01,
            help="控制正则化强度，值越大惩罚越重，特征选择越严格"
        )

        inputs['max_iter'] = st.number_input(
            "最大迭代次数",
            min_value=100,
            max_value=10000,
            value=1000,
            help="优化算法的最大迭代次数"
        )

        inputs['tol'] = st.number_input(
            "收敛容差",
            min_value=1e-6,
            max_value=1e-2,
            value=1e-4,
            format="%e",
            help="优化算法的收敛阈值"
        )

        # 随机种子
        inputs['random_state'] = st.number_input(
            "随机种子",
            min_value=0,
            max_value=100,
            value=42,
            help="设置随机种子以确保结果可重现"
        )

        # 选择是否拟合截距
        inputs['fit_intercept'] = st.checkbox(
            "拟合截距",
            value=True,
            help="是否计算模型的截距项"
        )

    with col2:
        st.subheader("参数说明")
        st.info("""
        **Lasso回归 参数指南:**

        - **alpha**: 最重要的参数，控制正则化强度
          - 较小的值：较弱正则化，更多特征被保留
          - 较大的值：较强正则化，更多特征系数变为0
        - **最大迭代次数**: 通常1000-5000足够收敛
        - **收敛容差**: 优化精度，值越小越精确但计算越慢

        **建议设置:**
        - alpha: 0.1-2.0之间开始尝试
        - 最大迭代次数: 1000-3000
        - 使用交叉验证选择最佳alpha值
        """)

        # 显示当前选择的参数
        st.write("当前参数设置:")
        st.write(f"- 正则化强度: {inputs['alpha']}")
        st.write(f"- 最大迭代次数: {inputs['max_iter']}")
        st.write(f"- 收敛容差: {inputs['tol']}")
        st.write(f"- 随机种子: {inputs['random_state']}")
        st.write(f"- 拟合截距: {'是' if inputs['fit_intercept'] else '否'}")

    return inputs, col2


if __name__ == "__main__":
    show()