import streamlit as st

# Define possible modelsd in a dict
# Format of the dict: model name -> model code

MODEL = {
    "model": "RandomForestRegressor",
}

# LightGBM can use -- categorical features -- as input directly. It doesn’t need to convert 
# to one-hot encoding, and is much faster than one-hot encoding (about 8x speed-up).

def show():
    """Shows the components for the template and returns user inputs as dict."""
    
    # `show()` is the only method required in this module. You can add any other code 
    # you like above or below. 
    
    inputs = {}  # dict to store all user inputs until return
    
    # with st.sidebar:
        
    # Render all template-specific sidebar components here. 

    # Use ## to denote sections. Common sections for training templates: 
    # Model, Input data, Preprocessing, Training, Visualizations
    # Store all user inputs in the `inputs` dict. This will be passed to the code
    # template later.
    inputs["model"] = MODEL["model"]
    
    # st.write("preprocessing")
    # inputs["Normalize"] = st.selectbox('normalize method', ['Z-Score Standardization','Min-Max Scale'])
    
    # st.info('TO SOLVE **REGRESSION**')
    
    # st.write('training')

    # st.write("No additional parameters")
    col1, col2 = st.columns([2,2])
    with col1:
        with st.expander("超参数配置"):

            inputs['criterion'] = st.selectbox('分裂准则',
                                               ('squared_error', 'friedman_mse', 'absolute_error', 'poisson'))
            inputs['nestimators'] = st.number_input('弱学习器数量', 1, 10000, 100)
            inputs['splitter'] = st.selectbox('分裂策略', ('best', 'random'))
            inputs['max depth'] = st.number_input('最大树深度', 1, 1000, 3)
            inputs['min samples leaf'] = st.number_input('叶节点最小样本数', 1, 100, 2)
            inputs['min samples split'] = st.number_input('内部节点分裂最小样本数', 2, 100, 3)
            inputs['oob score'] = False
            inputs['warm start'] = False
            inputs['njobs'] = None
            njob = st.checkbox('使用所有CPU核心', False)
            if njob:
                inputs['njobs'] = -1
            random_state = st.checkbox('固定随机种子(42)', True)
            if random_state:
                inputs['random state'] = 42
            else:
                inputs['random state'] = None

            auto_hyperparameters = st.checkbox('启用超参数自动优化', False)
            if auto_hyperparameters:
                inputs['auto hyperparameters'] = True
                inputs['init points'] = st.number_input('初始采样点数量', 1, 100, 10)
                inputs['iteration number'] = st.number_input('优化迭代次数', 1, 500, 10)
            else:
                inputs['auto hyperparameters'] = False
             # graph parameter
        # with st.expander("Unbalanced Data"):
        #     inputs['unbalanced data'] = st.checkbox('unbalanced data', False)
        #     if inputs['unbalanced data']:
        #         inputs['class weight'] = st.selectbox('class weight',(None,'balanced'))
        #         inputs['min weight fraction leaf'] = st.slider('min weight fraction leaf',0.0, 1.0, 0.0)
        #     else:
        #         inputs['class weight'] = None
        #         inputs['min weight fraction leaf'] = 0.0

    return inputs,col2


# To test the alg independent of the app or template, just run 
# `streamlit run alg.py` from within this folder.
if __name__ == "__main__":
    show()
    
    