import streamlit as st

MODEL = {
    "model": "RidgeRegressor",
}


def show():
    """Shows the components for the template and returns user inputs as dict."""

    inputs = {}  # dict to store all user inputs until return
    inputs["model"] = MODEL["model"]

    col1, col2 = st.columns([2, 2])

    with col1:
        st.subheader("岭回归 超参数设置")

        # 核心参数 - 正则化强度
        inputs['alpha'] = st.slider(
            "正则化强度 (alpha)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="控制L2正则化的强度，值越大惩罚越重，防止过拟合"
        )

        # 其他参数
        inputs['fit_intercept'] = st.checkbox(
            "拟合截距项",
            value=True,
            help="是否计算模型的截距项"
        )

        inputs['solver'] = st.selectbox(
            "求解器",
            ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
            index=0,
            help="用于计算的求解方法"
        )

        st.info("💡 岭回归是确定性算法，无需随机种子")

    with col2:
        st.subheader("参数说明")
        st.info("""
        **岭回归 参数指南:**

        - **alpha**: 最重要的参数，控制正则化强度
          - α=0: 退化为普通线性回归
          - α越大: 正则化越强，系数收缩越明显
        - **求解器**: 通常auto即可，大数据集可用sag或saga

        **建议设置:**
        - alpha: 通过交叉验证选择最佳值
        - 通常从0.1-10范围内尝试
        - 使用网格搜索寻找最优alpha
        """)

        # 显示当前选择的参数
        st.write("当前参数设置:")
        st.write(f"- 正则化强度: {inputs['alpha']}")
        st.write(f"- 拟合截距: {'是' if inputs['fit_intercept'] else '否'}")
        st.write(f"- 求解器: {inputs['solver']}")

    return inputs, col2


if __name__ == "__main__":
    show()