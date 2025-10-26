Bonding工艺数据分析与预测工具

基于Flask的Web应用，用于Bonding工艺数据的分析、模型训练和质量预测。本工具将原始Python脚本转换为用户友好的Web界面，支持数据上传、分析可视化、模型训练和预测功能。

功能特点

数据上传与分析：上传Excel数据文件，自动进行数据清洗和深度分析
可视化报告：生成多种图表展示工艺数据特征和关键指标
模型训练：基于分析数据训练XGBoost预测模型
质量预测：使用训练好的模型对新数据进行质量预测
高风险预警：识别高风险样本并生成预警报告

技术栈

后端：Python, Flask
前端：HTML, CSS, JavaScript, Bootstrap 5
数据处理：Pandas, NumPy
机器学习：Scikit-learn, XGBoost, Imbalanced-learn
可视化：Matplotlib, Seaborn

部署说明

本地开发

克隆仓库

bash
git clone https://github.com/yourusername/bonding-analysis-tool.git
cd bonding-analysis-tool


创建虚拟环境

bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate


安装依赖

bash
pip install -r requirements.txt


运行应用

bash
python app.py


在浏览器中访问 http://localhost:5000

部署到Render

将代码推送到GitHub仓库
在Render上创建新的Web服务，连接到GitHub仓库
设置构建命令：

bash
pip install -r requirements.txt


设置启动命令：

bash
gunicorn app:app


点击部署，等待部署完成

使用说明

在首页上传包含Bonding工艺数据的Excel文件
等待数据处理和模型训练完成（可能需要几分钟）
查看分析结果和模型性能指标
上传新数据进行预测
查看预测结果和高风险预警

数据格式要求

上传的Excel文件应包含以下关键列：

胶水条码
胶水编号
胶重结果
点胶路径
暴胶结果
压合结果
其他工艺参数列

数据应位于Excel文件的第一个工作表（Sheet1）。

项目结构

plaintext
bonding-analysis-tool/
├── app.py                 # Flask应用入口
├── requirements.txt       # 依赖包列表
├── Procfile               # Render部署配置
├── README.md              # 项目说明文档
├── static/                # 静态资源
│   ├── css/               # 样式表
│   ├── js/                # JavaScript文件
│   └── images/            # 生成的图表存放
├── templates/             # HTML模板
│   ├── index.html         # 主页面
│   ├── processing.html    # 处理中页面
│   ├── results.html       # 分析结果页面
│   └── prediction_results.html # 预测结果页面
├── uploads/               # 上传文件临时存储
├── models/                # 训练好的模型存储
└── static/results/        # 分析结果和图表存储


许可证

本项目采用MIT许可证 - 详情参见LICENSE文件