import streamlit as st

MODEL = {
    "model": "LinearRegressor",
}


def show():
    """Shows the components for the template and returns user inputs as dict."""

    inputs = {}  # dict to store all user inputs until return
    inputs["model"] = MODEL["model"]

    col1, col2 = st.columns([2, 2])

    with col1:
        st.subheader("线性回归 配置")

        # 线性回归的核心参数很少，主要是一些基础配置
        inputs['fit_intercept'] = st.checkbox(
            "拟合截距项",
            value=True,
            help="是否计算模型的截距（偏置项），通常建议保持开启"
        )

        # 可选的正则化（虽然标准线性回归没有，但可以提示用户）
        st.info("💡 线性回归通常不需要复杂参数调节")
        st.info("📊 模型性能主要取决于特征工程和数据质量")

    with col2:
        st.subheader("线性回归特点")
        st.info("""
        **算法特性:**

        ✅ **优点:**
        - 简单快速，计算效率高
        - 可解释性强，系数直接反映特征重要性
        - 适合作为基准模型
        - 对线性关系数据效果很好

        ⚠️ **局限性:**
        - 假设特征与目标呈线性关系
        - 对异常值敏感
        - 容易受到多重共线性影响
        - 无法处理复杂的非线性关系

        **适用场景:**
        - 特征与目标有明显的线性关系
        - 需要模型可解释性的场景
        - 作为其他复杂模型的基准对比
        """)

        # 显示当前选择的参数
        st.write("当前配置:")
        st.write(f"- 拟合截距: {'是' if inputs['fit_intercept'] else '否'}")
        st.write("- 其他参数: 使用默认值")

    return inputs, col2


# To test the alg independent of the app or template, just run
# `streamlit run alg.py` from within this folder.
if __name__ == "__main__":
    show()