import streamlit as st

# Define possible models in a dict
# Format of the dict: model name -> model code

MODEL = {
    "model": "RandomForestRegressor",
}


def show():
    """Shows the components for the template and returns user inputs as dict."""

    inputs = {}  # dict to store all user inputs until return
    inputs["model"] = MODEL["model"]

    col1, col2 = st.columns([2, 2])

    with col1:
        st.subheader("随机森林回归 超参数设置")

        # 核心参数
        inputs['nestimators'] = st.slider(
            "树的数量",
            min_value=10,
            max_value=1000,
            value=100,
            step=10,
            help="随机森林中决策树的数量，数量越多模型越稳定但计算量越大"
        )

        inputs['criterion'] = st.selectbox(
            '分裂标准',
            ('squared_error', 'friedman_mse', 'absolute_error', 'poisson'),
            index=0,
            help="衡量分裂质量的函数"
        )

        inputs['splitter'] = st.selectbox(
            '分裂策略',
            ('best', 'random'),
            index=0,
            help="选择分裂策略，best为最佳分裂，random为随机分裂"
        )

        # 树结构参数
        max_depth_check = st.checkbox('设置最大深度', False)
        if max_depth_check:
            inputs['max depth'] = st.number_input('最大深度', 1, 100, 10)
        else:
            inputs['max depth'] = None

        inputs['min samples split'] = st.number_input(
            '分裂最小样本数',
            min_value=2,
            max_value=100,
            value=2,
            help="内部节点再划分所需最小样本数"
        )

        inputs['min samples leaf'] = st.number_input(
            '叶节点最小样本数',
            min_value=1,
            max_value=50,
            value=1,
            help="叶节点最少需要的样本数"
        )

        # 其他参数
        inputs['njobs'] = -1 if st.checkbox('使用所有CPU核心', True) else 1
        inputs['random state'] = st.number_input(
            '随机种子',
            min_value=0,
            max_value=100,
            value=42,
            help="固定随机性，确保结果可重现"
        )

    with col2:
        st.subheader("参数说明")
        st.info("""
        **随机森林回归 参数指南:**

        - **树的数量**: 100-500之间通常效果较好
        - **分裂标准**: 
          - squared_error: 均方误差（默认）
          - absolute_error: 平均绝对误差
          - poisson: 泊松偏差
        - **最大深度**: 控制树复杂度，防止过拟合
        - **最小样本数**: 值越大越保守，防止过拟合

        **建议设置:**
        - 树数量: 100-300
        - 最大深度: 10-30
        - 最小样本分裂: 2-10
        - 最小样本叶节点: 1-5
        """)

        # 显示当前选择的参数
        st.write("当前参数设置:")
        st.write(f"- 树的数量: {inputs['nestimators']}")
        st.write(f"- 分裂标准: {inputs['criterion']}")
        st.write(f"- 分裂策略: {inputs['splitter']}")
        st.write(f"- 最大深度: {inputs.get('max depth', 'None')}")
        st.write(f"- 分裂最小样本数: {inputs['min samples split']}")
        st.write(f"- 叶节点最小样本数: {inputs['min samples leaf']}")
        st.write(f"- 使用所有CPU: {'是' if inputs['njobs'] else '否'}")
        st.write(f"- 随机种子: {inputs['random state']}")

    return inputs, col2


# To test the alg independent of the app or template, just run
# `streamlit run alg.py` from within this folder.
if __name__ == "__main__":
    show()