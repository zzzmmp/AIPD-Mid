# 在代码开头添加以下代码来检查PATH
import os
import graphviz
import streamlit as st

# 打印当前的PATH环境变量
st.write("当前PATH:", os.environ["PATH"])

# 手动添加Graphviz的安装路径
graphviz_bin_path = r"D:\Graphviz\bin"  # 根据实际安装路径修改
if graphviz_bin_path not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + graphviz_bin_path
    st.write("已手动添加Graphviz路径到PATH")