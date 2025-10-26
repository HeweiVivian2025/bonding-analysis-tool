from flask import Flask, render_template, request, redirect, url_for, flash, session
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适合Web环境
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.graph_objects as go
import json
import joblib
import uuid
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shutil
from werkzeug.utils import secure_filename

# 初始化Flask应用
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev_secret_key')  # 生产环境应使用环境变量
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB上传限制
app.config['ALLOWED_EXTENSIONS'] = {'xlsx'}
app.config['RESULT_FOLDER'] = 'static/results'
app.config['MODEL_FOLDER'] = 'models'

# 确保必要目录存在
for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULT_FOLDER'], app.config['MODEL_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# 全局变量存储分析和训练状态
task_status = {}

# 辅助函数：检查文件扩展名
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# 辅助函数：生成唯一任务ID
def generate_task_id():
    return str(uuid.uuid4())[:8]

# 辅助函数：清除过期文件
def clean_old_files():
    # 实际应用中应实现文件清理逻辑
    pass

# 数据处理函数 - 从原始代码提取并修改
def process_data(file_path, task_id):
    """处理上传的数据，执行第一段代码的分析功能"""
    result_path = os.path.join(app.config['RESULT_FOLDER'], task_id)
    os.makedirs(result_path, exist_ok=True)
    
    try:
        # 读取原始数据
        df = pd.read_excel(file_path, sheet_name='Sheet1')
        df.to_csv(os.path.join(result_path, 'raw_data.csv'), index=False)
        
        # 标准化结果字段
        cat_cols_all = ['胶水条码', '胶水编号', '线别', '点胶机编号', '点胶路径', '治具号', 
                       '压合机台号', '其他信息', '镭雕号', '暴胶结果', '胶重结果', '压合结果']
        for col in cat_cols_all:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
                df[col] = df[col].replace(['NAN', 'NONE', 'NULL', '', 'NA'], np.nan)
        
        # 重算压合结果
        def recalculate_final_bonding_result(row):
            if (row['胶重结果'] == 'NG' or row['点胶路径'] == 'NG' or 
                row['暴胶结果'] == 'NG' or row['压合结果'] == 'NG'):
                return 'NG'
            required = ['胶重结果', '点胶路径', '暴胶结果', '压合结果']
            if any(pd.isna(row[col]) for col in required):
                return 'NG'
            return 'OK'
        
        df['压合结果_重算'] = df.apply(recalculate_final_bonding_result, axis=1)
        df['压合结果_数值'] = df['压合结果_重算'].map({'OK': 0, 'NG': 1})
        
        # 时间列处理
        df['CCD时间'] = pd.to_datetime(df['CCD时间'], errors='coerce')
        
        # 条件性填充
        df_filled = df.copy()
        fill_cols_1 = ['胶水条码', '胶水编号', '线别', '点胶机编号', '点胶时间', '胶重']
        fill_cols_2 = ['CCD时间']
        fill_cols_3 = ['胶水暴露时间']
        fill_cols_4 = [col for col in df.columns if col not in ['其他信息', '胶重结果', '点胶路径', 
                                                               '暴胶结果', '压合结果', '压合结果_重算', 
                                                               '压合结果_数值', '检测阶段', '胶重结果_状态', 
                                                               '点胶路径_状态', '暴胶结果_状态', 'CCD时间'] 
                      and col not in fill_cols_1 + fill_cols_2 + fill_cols_3]
        
        mask_jiaozhong_ok = df_filled['胶重结果'] == 'OK'
        df_filled.loc[mask_jiaozhong_ok, fill_cols_1] = df_filled.loc[mask_jiaozhong_ok, fill_cols_1].fillna(method='ffill')
        
        mask_dianjiao_ok = (df_filled['胶重结果'] == 'OK') & (df_filled['点胶路径'] == 'OK')
        df_filled.loc[mask_dianjiao_ok, fill_cols_2] = df_filled.loc[mask_dianjiao_ok, fill_cols_2].fillna(method='ffill')
        
        mask_baogao_ok = (df_filled['胶重结果'] == 'OK') & (df_filled['点胶路径'] == 'OK') & (df_filled['暴胶结果'] == 'OK')
        df_filled.loc[mask_baogao_ok, fill_cols_3] = df_filled.loc[mask_baogao_ok, fill_cols_3].fillna(method='ffill')
        
        mask_all_ok = (df_filled['胶重结果'] == 'OK') & (df_filled['点胶路径'] == 'OK') & (df_filled['暴胶结果'] == 'OK') & (df_filled['压合结果'] == 'OK')
        df_filled.loc[mask_all_ok, fill_cols_4] = df_filled.loc[mask_all_ok, fill_cols_4].fillna(method='ffill')
        
        # 状态标记
        def mark_skip_reason_jiaozhong(row):
            if pd.isna(row['胶重结果']):
                return 'Missing_Unknown'
            else:
                return 'Measured_' + row['胶重结果']
        
        def mark_skip_reason_dianjiao_path(row):
            if pd.isna(row['点胶路径']):
                if row['胶重结果'] == 'NG':
                    return 'Skipped_due_to_胶重NG'
                else:
                    return 'Missing_Unknown'
            else:
                return 'Measured_' + row['点胶路径']
        
        def mark_skip_reason_baogao(row):
            if pd.isna(row['暴胶结果']):
                if row['胶重结果'] == 'NG':
                    return 'Skipped_due_to_胶重NG'
                elif row['点胶路径'] == 'NG':
                    return 'Skipped_due_to_点胶路径NG'
                else:
                    return 'Missing_Unknown'
            else:
                return 'Measured_' + row['暴胶结果']
        
        df_filled['胶重结果_状态'] = df_filled.apply(mark_skip_reason_jiaozhong, axis=1)
        df_filled['点胶路径_状态'] = df_filled.apply(mark_skip_reason_dianjiao_path, axis=1)
        df_filled['暴胶结果_状态'] = df_filled.apply(mark_skip_reason_baogao, axis=1)
        
        df_filled['检测阶段'] = np.select(
            [
                df_filled['胶重结果'].isna(),
                df_filled['胶重结果'] == 'NG',
                df_filled['点胶路径'] == 'NG',
                df_filled['暴胶结果'].notna(),
            ],
            [
                '未开始检测',
                '仅完成胶重',
                '完成胶重+点胶路径',
                '完成全部三项'
            ],
            default='完成胶重+点胶路径'
        )
        
        # 衍生变量与时间特征
        df_filled['检测小时'] = df_filled['CCD时间'].dt.hour
        df_filled['是否工作日'] = (df_filled['CCD时间'].dt.dayofweek < 5).astype(int)
        df_filled['班次'] = np.where(df_filled['检测小时'].between(8, 20), '白班', '夜班')
        df_filled['加热板温差'] = abs(df_filled['上/下加热板设定温度'] - df_filled['上/下加热板温度'])
        df_filled['单位点胶速率'] = df_filled['胶重'] / (df_filled['点胶时间'] + 1e-6)
        
        # 保存填充后数据
        filled_data_path = os.path.join(result_path, 'filled_data.csv')
        df_filled.to_csv(filled_data_path, index=False)
        
        # 生成可视化图表
        visualizations = []
        
        # 压合结果分布图
        plt.figure(figsize=(10, 6))
        sns.countplot(x='压合结果_数值', data=df_filled)
        plt.title('压合结果分布')
        plt.ylabel('数量')
        plt.xlabel('压合结果 (0=OK, 1=NG)')
        dist_plot_path = os.path.join(result_path, 'pressure_result_distribution.png')
        plt.savefig(dist_plot_path)
        plt.close()
        visualizations.append({
            'title': '压合结果分布',
            'path': os.path.relpath(dist_plot_path, app.static_folder),
            'description': '展示压合结果的分布情况，0表示OK，1表示NG'
        })
        
        # 时间序列分析
        if 'CCD时间' in df_filled.columns and not df_filled['CCD时间'].isna().all():
            df_time = df_filled.copy()
            df_time['日期'] = df_time['CCD时间'].dt.date
            daily_ng = df_time.groupby('日期')['压合结果_数值'].mean().reset_index()
            
            plt.figure(figsize=(12, 6))
            plt.plot(daily_ng['日期'], daily_ng['压合结果_数值'], marker='o')
            plt.title('每日NG率趋势')
            plt.ylabel('NG率')
            plt.xlabel('日期')
            plt.xticks(rotation=45)
            plt.tight_layout()
            trend_plot_path = os.path.join(result_path, 'daily_ng_trend.png')
            plt.savefig(trend_plot_path)
            plt.close()
            visualizations.append({
                'title': '每日NG率趋势',
                'path': os.path.relpath(trend_plot_path, app.static_folder),
                'description': '展示每日压合NG率的变化趋势'
            })
        
        # 相关性热图
        numeric_cols = ['胶重', '胶水暴露时间', '点胶时间', '上/下加热板设定温度', 
                       '上/下加热板温度', '热压气压', '热压时间', '胶层温度', '放置时间',
                       '加热板温差', '单位点胶速率', '压合结果_数值']
        numeric_df = df_filled[numeric_cols].select_dtypes(include=['float64', 'int64'])
        
        plt.figure(figsize=(12, 10))
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('特征相关性热图')
        plt.tight_layout()
        corr_plot_path = os.path.join(result_path, 'feature_correlation.png')
        plt.savefig(corr_plot_path)
        plt.close()
        visualizations.append({
            'title': '特征相关性热图',
            'path': os.path.relpath(corr_plot_path, app.static_folder),
            'description': '展示各数值特征之间的相关性'
        })
        
        # 保存可视化结果
        with open(os.path.join(result_path, 'visualizations.json'), 'w') as f:
            json.dump(visualizations, f)
            
        return {
            'status': 'success',
            'message': '数据处理完成',
            'result_path': result_path,
            'visualizations': visualizations
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'数据处理失败: {str(e)}'
        }

# 模型训练函数 - 从第二段代码提取并修改
def train_model(data_path, task_id):
    """训练预测模型，执行第二段代码的功能"""
    model_path = os.path.join(app.config['MODEL_FOLDER'], task_id)
    os.makedirs(model_path, exist_ok=True)
    
    try:
        # 加载处理后的数据
        df = pd.read_csv(os.path.join(data_path, 'filled_data.csv'))
        
        # 确保必要的列存在
        required_columns = ['压合结果_数值', '点胶时间', '胶水暴露时间', '单位点胶速率', 
                           '检测小时', '加热板温差', '放置时间', '班次', '是否工作日']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"数据缺少必要的列: {', '.join(missing_cols)}")
        
        # 特征工程
        df['班次_encoded'] = df['班次'].map({'白班': 0, '夜班': 1})
        
        numeric_features = [
            '点胶时间', '胶水暴露时间', '单位点胶速率', '检测小时', 
            '加热板温差', '放置时间'
        ]
        
        categorical_features = [
            '班次_encoded', '是否工作日'
        ]
        
        # 准备特征和目标变量
        X = df[numeric_features + categorical_features]
        y = df['压合结果_数值']
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 处理类别不平衡
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        # 标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_resampled)
        X_test_scaled = scaler.transform(X_test)
        
        # 训练XGBoost模型
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        xgb_model.fit(X_train_scaled, y_train_resampled)
        
        # 评估模型
        y_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
        y_pred = xgb_model.predict(X_test_scaled)
        
        # 计算评估指标
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # 保存模型和scaler
        joblib.dump(xgb_model, os.path.join(model_path, 'xgb_model.pkl'))
        joblib.dump(scaler, os.path.join(model_path, 'scaler.pkl'))
        joblib.dump(numeric_features, os.path.join(model_path, 'numeric_features.pkl'))
        joblib.dump(categorical_features, os.path.join(model_path, 'categorical_features.pkl'))
        
        # 生成特征重要性图
        plt.figure(figsize=(10, 8))
        feature_importance = pd.DataFrame({
            'feature': numeric_features + categorical_features,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('特征重要性')
        plt.tight_layout()
        importance_plot_path = os.path.join(data_path, 'feature_importance.png')
        plt.savefig(importance_plot_path)
        plt.close()
        
        return {
            'status': 'success',
            'message': '模型训练完成',
            'model_path': model_path,
            'metrics': metrics,
            'feature_importance_path': os.path.relpath(importance_plot_path, app.static_folder)
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'模型训练失败: {str(e)}'
        }

# 预测函数
def predict_data(file_path, model_task_id, task_id):
    """使用训练好的模型进行预测"""
    result_path = os.path.join(app.config['RESULT_FOLDER'], task_id)
    os.makedirs(result_path, exist_ok=True)
    
    try:
        # 加载模型和相关文件
        model_path = os.path.join(app.config['MODEL_FOLDER'], model_task_id)
        if not os.path.exists(model_path):
            raise ValueError(f"模型任务ID不存在: {model_task_id}")
            
        xgb_model = joblib.load(os.path.join(model_path, 'xgb_model.pkl'))
        scaler = joblib.load(os.path.join(model_path, 'scaler.pkl'))
        numeric_features = joblib.load(os.path.join(model_path, 'numeric_features.pkl'))
        categorical_features = joblib.load(os.path.join(model_path, 'categorical_features.pkl'))
        
        # 加载预测数据
        df = pd.read_excel(file_path, sheet_name='Sheet1')
        
        # 检查是否有ID列
        if 'ID' not in df.columns:
            raise ValueError("预测数据必须包含'ID'列")
            
        # 数据预处理
        if '班次' in df.columns:
            df['班次_encoded'] = df['班次'].map({'白班': 0, '夜班': 1})
        
        # 检查必要的特征是否存在
        required_features = numeric_features + categorical_features
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            raise ValueError(f"预测数据缺少必要的特征: {', '.join(missing_features)}")
        
        # 准备特征
        X = df[required_features]
        
        # 处理缺失值
        X = X.fillna(X.median(numeric_only=True))
        
        # 标准化
        X_scaled = scaler.transform(X)
        
        # 预测
        y_pred_proba = xgb_model.predict_proba(X_scaled)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # 生成预测结果
        df['NG概率'] = y_pred_proba
        df['预测结果'] = y_pred
        df['预测结果'] = df['预测结果'].map({0: 'OK', 1: 'NG'})
        
        # 高风险预警 (NG概率 > 0.8)
        high_risk = df[df['NG概率'] > 0.8].copy()
        
        # 保存结果
        prediction_result_path = os.path.join(result_path, 'prediction_results.csv')
        df.to_csv(prediction_result_path, index=False)
        
        high_risk_path = os.path.join(result_path, 'high_risk_alerts.csv')
        high_risk.to_csv(high_risk_path, index=False)
        
        # 生成预测结果可视化
        plt.figure(figsize=(10, 6))
        sns.histplot(y_pred_proba, bins=20, kde=True)
        plt.axvline(x=0.5, color='r', linestyle='--', label='分类阈值 (0.5)')
        plt.axvline(x=0.8, color='orange', linestyle='--', label='高风险阈值 (0.8)')
        plt.title('预测NG概率分布')
        plt.xlabel('NG概率')
        plt.ylabel('样本数')
        plt.legend()
        pred_dist_path = os.path.join(result_path, 'prediction_distribution.png')
        plt.savefig(pred_dist_path)
        plt.close()
        
        return {
            'status': 'success',
            'message': '预测完成',
            'result_path': result_path,
            'predictions': df[['ID', '预测结果', 'NG概率']].to_dict('records'),
            'high_risk_count': len(high_risk),
            'prediction_distribution': os.path.relpath(pred_dist_path, app.static_folder),
            'high_risk_path': os.path.relpath(high_risk_path, app.static_folder)
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'预测失败: {str(e)}'
        }

# 路由：首页
@app.route('/')
def index():
    return render_template('index.html')

# 路由：上传数据进行分析
@app.route('/upload_data', methods=['POST'])
def upload_data():
    clean_old_files()
    
    # 检查是否有文件上传
    if 'data_file' not in request.files:
        flash('没有文件部分', 'danger')
        return redirect(url_for('index'))
    
    file = request.files['data_file']
    
    # 如果用户没有选择文件
    if file.filename == '':
        flash('没有选择文件', 'danger')
        return redirect(url_for('index'))
    
    # 如果文件合法
    if file and allowed_file(file.filename):
        task_id = generate_task_id()
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{filename}")
        file.save(file_path)
        
        # 异步处理数据
        from threading import Thread
        thread = Thread(target=lambda: process_and_train(task_id, file_path))
        thread.start()
        
        task_status[task_id] = {
            'status': 'processing',
            'step': 'data_processing',
            'message': '正在处理数据...',
            'start_time': datetime.now().isoformat()
        }
        
        return render_template('processing.html', task_id=task_id)
    
    flash('不支持的文件格式，仅支持.xlsx文件', 'danger')
    return redirect(url_for('index'))

# 辅助函数：处理数据并训练模型（异步执行）
def process_and_train(task_id, file_path):
    # 处理数据
    task_status[task_id] = {
        'status': 'processing',
        'step': 'data_processing',
        'message': '正在处理数据...',
        'progress': 30
    }
    
    process_result = process_data(file_path, task_id)
    
    if process_result['status'] != 'success':
        task_status[task_id] = {
            'status': 'error',
            'step': 'data_processing',
            'message': process_result['message']
        }
        return
    
    # 训练模型
    task_status[task_id] = {
        'status': 'processing',
        'step': 'model_training',
        'message': '正在训练模型...',
        'progress': 70,
        'result_path': process_result['result_path']
    }
    
    train_result = train_model(process_result['result_path'], task_id)
    
    if train_result['status'] != 'success':
        task_status[task_id] = {
            'status': 'error',
            'step': 'model_training',
            'message': train_result['message'],
            'result_path': process_result['result_path']
        }
        return
    
    # 完成
    task_status[task_id] = {
        'status': 'completed',
        'step': 'done',
        'message': '数据处理和模型训练已完成',
        'progress': 100,
        'result_path': process_result['result_path'],
        'model_path': train_result['model_path'],
        'metrics': train_result['metrics'],
        'visualizations': process_result['visualizations'],
        'feature_importance_path': train_result['feature_importance_path']
    }

# 路由：检查任务状态
@app.route('/task_status/<task_id>')
def check_status(task_id):
    status = task_status.get(task_id, {'status': 'not_found', 'message': '任务ID不存在'})
    return json.dumps(status)

# 路由：查看分析和模型结果
@app.route('/results/<task_id>')
def show_results(task_id):
    status = task_status.get(task_id)
    if not status or status['status'] != 'completed':
        return redirect(url_for('index'))
    
    return render_template('results.html', 
                          task_id=task_id, 
                          status=status)

# 路由：上传预测数据
@app.route('/upload_prediction/<model_task_id>', methods=['GET', 'POST'])
def upload_prediction(model_task_id):
    if request.method == 'GET':
        return render_template('predict.html', model_task_id=model_task_id)
    
    # POST请求处理
    if 'prediction_file' not in request.files:
        flash('没有文件部分', 'danger')
        return redirect(url_for('upload_prediction', model_task_id=model_task_id))
    
    file = request.files['prediction_file']
    
    if file.filename == '':
        flash('没有选择文件', 'danger')
        return redirect(url_for('upload_prediction', model_task_id=model_task_id))
    
    if file and allowed_file(file.filename):
        task_id = generate_task_id()
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"pred_{task_id}_{filename}")
        file.save(file_path)
        
        # 异步执行预测
        from threading import Thread
        thread = Thread(target=lambda: run_prediction(model_task_id, task_id, file_path))
        thread.start()
        
        task_status[task_id] = {
            'status': 'processing',
            'step': 'predicting',
            'message': '正在进行预测...',
            'start_time': datetime.now().isoformat()
        }
        
        return render_template('predict_processing.html', task_id=task_id, model_task_id=model_task_id)
    
    flash('不支持的文件格式，仅支持.xlsx文件', 'danger')
    return redirect(url_for('upload_prediction', model_task_id=model_task_id))

# 辅助函数：执行预测（异步）
def run_prediction(model_task_id, task_id, file_path):
    task_status[task_id] = {
        'status': 'processing',
        'step': 'predicting',
        'message': '正在进行预测...',
        'progress': 50
    }
    
    predict_result = predict_data(file_path, model_task_id, task_id)
    
    if predict_result['status'] != 'success':
        task_status[task_id] = {
            'status': 'error',
            'step': 'predicting',
            'message': predict_result['message']
        }
        return
    
    # 获取模型训练时的可视化结果
    model_status = task_status.get(model_task_id, {})
    
    task_status[task_id] = {
        'status': 'completed',
        'step': 'prediction_done',
        'message': '预测已完成',
        'progress': 100,
        'result_path': predict_result['result_path'],
        'predictions': predict_result['predictions'],
        'high_risk_count': predict_result['high_risk_count'],
        'prediction_distribution': predict_result['prediction_distribution'],
        'high_risk_path': predict_result['high_risk_path'],
        'model_visualizations': model_status.get('visualizations', [])
    }

# 路由：查看预测结果
@app.route('/prediction_results/<task_id>')
def show_prediction_results(task_id):
    status = task_status.get(task_id)
    if not status or status['status'] != 'completed':
        return redirect(url_for('index'))
    
    return render_template('prediction_results.html', 
                          task_id=task_id, 
                          status=status)

# 路由：下载结果文件
@app.route('/download/<path:filename>')
def download_file(filename):
    from flask import send_from_directory
    return send_from_directory(app.static_folder, 
                              filename, as_attachment=True)

# 应用入口
if __name__ == '__main__':
    app.run(debug=True)